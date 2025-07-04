import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.utils
from flask import Flask, render_template, jsonify, request, send_from_directory
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from evaluator.scorers.base import ScorerResult


class WebDashboard:
    
    def __init__(self, port: int = 5000, host: str = "127.0.0.1", enable_tunnel: bool = True, tunnel_token: Optional[str] = None):
        self.port = port
        self.host = host
        self.enable_tunnel = enable_tunnel
        self.tunnel_token = tunnel_token or os.environ.get("CLOUDFLARE_TUNNEL_TOKEN")
        self.frontend_process = None
        self.backend_process = None
        self.tunnel_process = None
        self.public_url = None
        self.dashboard_path = Path(__file__).parent
    
    def load_results(self, results: Dict[str, Dict[str, ScorerResult]], trajectories: Optional[Dict] = None):
        self._results_for_backend = results
        self._trajectories_for_backend = trajectories or {}
    
    def start_server(self, open_browser: bool = True, debug: bool = False, production: bool = True):
        print(f"Starting web dashboard at http://{self.host}:{self.port}")
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        self._ensure_frontend_ready()
        
        print("Starting backend with data, then frontend...")
        self._start_backend_with_data(debug)
        
        if production:
            self._start_frontend_static()
        else:
            self._start_frontend_dev()
        
        if self.enable_tunnel:
            self._start_tunnel()
        
        self._display_urls()
        
        if open_browser:
            def open_browser_delayed():
                time.sleep(3)
                url_to_open = self.public_url if self.public_url else f'http://{self.host}:{self.port}'
                webbrowser.open(url_to_open)
            
            browser_thread = threading.Thread(target=open_browser_delayed)
            browser_thread.daemon = True
            browser_thread.start()
        
        self._wait_for_shutdown()
    
    def _start_tunnel(self):
        try:
            subprocess.run(["cloudflared", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("cloudflared not found. Falling back to localhost.run tunnel...")
            self._start_localhost_run_tunnel()
            return
        
        if self.tunnel_token:
            print("Creating Cloudflare tunnel with provided token...")
            self._start_cloudflare_named_tunnel()
        else:
            print("Creating Cloudflare quick tunnel (temporary URL)...")
            self._start_cloudflare_quick_tunnel()
    
    def _start_cloudflare_named_tunnel(self):
        tunnel_cmd = [
            "cloudflared", "tunnel", "run", 
            "--token", self.tunnel_token
        ]
        
        print(f"Starting named Cloudflare tunnel...")
        
        try:
            self.tunnel_process = subprocess.Popen(
                tunnel_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            def monitor_cloudflare_tunnel():
                try:
                    start_time = time.time()
                    timeout = 30
                    
                    while time.time() - start_time < timeout:
                        if self.tunnel_process.poll() is not None:
                            print("Tunnel process exited")
                            _, stderr = self.tunnel_process.communicate()
                            if stderr:
                                print(f"Tunnel error: {stderr}")
                            break
                            
                        try:
                            if self.tunnel_process.stdout.readable():
                                line = self.tunnel_process.stdout.readline()
                                if line:
                                    line = line.strip()
                                    print(f"Tunnel: {line}")
                                    
                                    if "https://" in line and any(domain in line for domain in [".cfargotunnel.com", ".trycloudflare.com"]):
                                        url_match = re.search(r'https://[^\s]+\.(?:cfargotunnel\.com|trycloudflare\.com)', line)
                                        if url_match:
                                            self.public_url = url_match.group(0)
                                            print(f"Cloudflare tunnel URL: {self.public_url}")
                                            return
                                    
                                    if "serving at" in line.lower() or "started tunnel" in line.lower():
                                        print("Cloudflare tunnel started successfully!")
                                        print("Check your Cloudflare Zero Trust dashboard for the public URL")
                                        return
                                        
                        except Exception as e:
                            print(f"Error reading tunnel output: {e}")
                            time.sleep(1)
                            continue
                        
                        time.sleep(1)
                    
                    if not self.public_url:
                        print("Timeout waiting for tunnel URL")
                        print("Check your Cloudflare Zero Trust dashboard to see the tunnel status")
                        
                except Exception as e:
                    print(f"Error monitoring tunnel: {e}")
            
            tunnel_thread = threading.Thread(target=monitor_cloudflare_tunnel)
            tunnel_thread.daemon = True
            tunnel_thread.start()
            
            time.sleep(5)
            
        except Exception as e:
            print(f"Failed to start Cloudflare tunnel: {e}")
            print("Make sure your tunnel token is valid and cloudflared is installed")
            self.tunnel_process = None
    
    def _start_cloudflare_quick_tunnel(self):
        tunnel_cmd = [
            "cloudflared", "tunnel", 
            "--url", f"http://localhost:{self.port}"
        ]
        
        print(f"Starting Cloudflare quick tunnel...")
        
        try:
            self.tunnel_process = subprocess.Popen(
                tunnel_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            def monitor_quick_tunnel():
                try:
                    start_time = time.time()
                    timeout = 30
                    
                    while time.time() - start_time < timeout:
                        if self.tunnel_process.poll() is not None:
                            print("Tunnel process exited")
                            break
                            
                        try:
                            line = self.tunnel_process.stderr.readline()
                            if line:
                                line = line.strip()
                                if line:
                                    print(f"Tunnel: {line}")
                                    
                                    url_match = re.search(r'https://[a-zA-Z0-9\-]+\.trycloudflare\.com', line)
                                    if url_match:
                                        self.public_url = url_match.group(0)
                                        print(f"Cloudflare quick tunnel URL: {self.public_url}")
                                        return
                                        
                        except Exception as e:
                            time.sleep(1)
                            continue
                        
                        time.sleep(1)
                    
                    if not self.public_url:
                        print("Timeout waiting for tunnel URL")
                        
                except Exception as e:
                    print(f"Error monitoring tunnel: {e}")
            
            tunnel_thread = threading.Thread(target=monitor_quick_tunnel)
            tunnel_thread.daemon = True
            tunnel_thread.start()
            
            time.sleep(5)
            
        except Exception as e:
            print(f"Failed to start Cloudflare quick tunnel: {e}")
            print("Falling back to localhost.run...")
            self._start_localhost_run_tunnel()
    
    def _start_localhost_run_tunnel(self):
        print("Creating fallback tunnel via localhost.run...")
        
        tunnel_port = self.port
        
        simple_tunnel_cmd = [
            "ssh", "-R", f"80:localhost:{tunnel_port}", 
            "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null",
            "-T",
            "nokey@localhost.run"
        ]
        
        print(f"Running: {' '.join(simple_tunnel_cmd)}")
        
        try:
            self.tunnel_process = subprocess.Popen(
                simple_tunnel_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1
            )
            
            def monitor_tunnel():
                try:
                    import select
                    import time
                    
                    start_time = time.time()
                    timeout = 30
                    
                    while time.time() - start_time < timeout:
                        if self.tunnel_process.poll() is not None:
                            print("Tunnel process exited")
                            break
                            
                        try:
                            ready, _, _ = select.select([self.tunnel_process.stdout, self.tunnel_process.stderr], [], [], 2)
                        except:
                            continue
                        
                        for stream in ready:
                            try:
                                line = stream.readline()
                                if line:
                                    line = line.strip()
                                    if line:
                                        print(f"Tunnel output: {line}")
                                        
                                        domain_matches = re.findall(r'([a-zA-Z0-9\-]+\.(?:localhost\.run|lhr\.life))', line)
                                        for domain in domain_matches:
                                            if 'admin' not in domain and 'nokey' not in domain:
                                                self.public_url = f"https://{domain}"
                                                print(f"Public URL found: {self.public_url}")
                                                return
                                        
                                        https_matches = re.findall(r'(https://[a-zA-Z0-9\-]+\.(?:localhost\.run|lhr\.life))', line)
                                        for url in https_matches:
                                            if 'admin' not in url:
                                                self.public_url = url
                                                print(f"Public URL found: {self.public_url}")
                                                return
                            except:
                                continue
                    
                    if not self.public_url:
                        print("Timeout waiting for tunnel URL")
                        
                except Exception as e:
                    print(f"Error monitoring tunnel: {e}")
            
            tunnel_thread = threading.Thread(target=monitor_tunnel)
            tunnel_thread.daemon = True
            tunnel_thread.start()
            
            time.sleep(3)
            
        except FileNotFoundError:
            print("SSH not found. Public tunnel not available.")
            print("Install SSH to enable public tunneling via localhost.run")
            self.tunnel_process = None
        except Exception as e:
            print(f"Failed to start tunnel: {e}")
            self.tunnel_process = None
    
    def _display_urls(self):
        print("\n" + "="*60)
        print("DASHBOARD ACCESS URLS")
        print("="*60)
        print(f"Local:  http://{self.host}:{self.port}")
        if self.public_url:
            print(f"Public: {self.public_url}")
        print("="*60)
        print("Press Ctrl+C to stop the server")
        print()
    
    def _ensure_frontend_ready(self):
        frontend_dir = self.dashboard_path
        
        if not (frontend_dir / "node_modules").exists():
            print("Installing frontend dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"], 
                    cwd=frontend_dir, 
                    check=True,
                    capture_output=True
                )
                print("Frontend dependencies installed")
            except subprocess.CalledProcessError as e:
                print(f"Failed to install frontend dependencies: {e}")
                print("Make sure Node.js and npm are installed")
                raise
    
    def _start_backend_with_data(self, debug: bool = False):
        backend_port = self.port + 1000
        
        import subprocess
        import os
        import tempfile
        import pickle
        
        project_root = Path.cwd()
        
        data_file = None
        if hasattr(self, '_results_for_backend') and self._results_for_backend:
            print("Preparing data for backend...")
            
            data_file = tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl')
            
            data_to_transfer = {
                'results': self._results_for_backend,
                'trajectories': self._trajectories_for_backend
            }
            
            pickle.dump(data_to_transfer, data_file)
            data_file.close()
            
            print(f"Data file: {data_file.name}")
        
        backend_cmd = [
            "bash", "-c", 
            f"cd {project_root} && "
            f"DASHBOARD_DATA_FILE={data_file.name if data_file else ''} "
            f"python -m uvicorn dashboard.backend.main:app "
            f"--host {self.host} --port {backend_port} "
            f"{'--reload' if debug else ''} "
            f"--log-level info"
        ]
        
        print(f"Starting backend server with data...")
        print(f"Working directory: {project_root}")
        print(f"Backend will be at: http://{self.host}:{backend_port}")
        
        try:
            self.backend_process = subprocess.Popen(
                backend_cmd,
                cwd=project_root
            )
            
            time.sleep(3)
            
            if self.backend_process.poll() is None:
                print(f"Backend API running at http://{self.host}:{backend_port}")
            else:
                print("Backend process exited immediately - check the error messages above")
                
        except Exception as e:
            print(f"Failed to start backend: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if data_file:
                self._temp_data_file = data_file.name
    
    def _start_frontend_dev(self):
        frontend_dir = self.dashboard_path
        
        def run_frontend():
            self.frontend_process = subprocess.Popen(
                ["npm", "run", "dev", "--", "--host", self.host, "--port", str(self.port)],
                cwd=frontend_dir
            )
            
            while True:
                if self.frontend_process.poll() is not None:
                    break
                time.sleep(1)
            
        frontend_thread = threading.Thread(target=run_frontend)
        frontend_thread.daemon = True
        frontend_thread.start()
        
        time.sleep(3)
        print(f"Frontend development server running at http://{self.host}:{self.port}")
    
    def _start_frontend_static(self):
        frontend_dir = self.dashboard_path
        build_dir = frontend_dir / "build"
        
        if not build_dir.exists():
            print("Build directory not found. Building frontend...")
            subprocess.run(["npm", "run", "build"], cwd=frontend_dir, check=True)
        
        def run_static_server():
            import http.server
            import socketserver
            import os
            import urllib.request
            import urllib.parse
            
            os.chdir(build_dir)
            
            backend_port = self.port + 1000
            
            class SvelteKitHandler(http.server.SimpleHTTPRequestHandler):
                def end_headers(self):
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    super().end_headers()
                
                def do_GET(self):
                    if self.path.startswith('/api'):
                        self._proxy_to_backend()
                        return
                    
                    if not os.path.exists(self.path.lstrip('/')):
                        self.path = '/index.html'
                    return super().do_GET()
                
                def do_POST(self):
                    if self.path.startswith('/api'):
                        self._proxy_to_backend()
                        return
                    
                    self.send_error(404)
                
                def do_DELETE(self):
                    if self.path.startswith('/api'):
                        self._proxy_to_backend()
                        return
                    
                    self.send_error(404)
                
                def do_OPTIONS(self):
                    self.send_response(200)
                    self.send_header('Access-Control-Allow-Origin', '*')
                    self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS, DELETE')
                    self.send_header('Access-Control-Allow-Headers', 'Content-Type')
                    self.end_headers()
                
                def _proxy_to_backend(self):
                    try:
                        backend_url = f"http://{self.server.server_address[0]}:{backend_port}{self.path}"
                        
                        content_length = int(self.headers.get('content-length', 0))
                        post_data = self.rfile.read(content_length) if content_length > 0 else None
                        
                        req = urllib.request.Request(
                            backend_url,
                            data=post_data,
                            method=self.command
                        )
                        
                        for header, value in self.headers.items():
                            if header.lower() not in ['host', 'content-length']:
                                req.add_header(header, value)
                        
                        try:
                            with urllib.request.urlopen(req, timeout=30) as response:
                                self.send_response(response.getcode())
                                
                                for header, value in response.headers.items():
                                    if header.lower() != 'transfer-encoding':
                                        self.send_header(header, value)
                                
                                self.end_headers()
                                
                                self.wfile.write(response.read())
                                
                        except urllib.error.HTTPError as e:
                            self.send_response(e.code)
                            for header, value in e.headers.items():
                                if header.lower() != 'transfer-encoding':
                                    self.send_header(header, value)
                            self.end_headers()
                            self.wfile.write(e.read())
                            
                    except Exception as e:
                        print(f"Proxy error: {e}")
                        self.send_error(502, f"Backend proxy error: {e}")
            
            with socketserver.TCPServer((self.host, self.port), SvelteKitHandler) as httpd:
                print(f"Static frontend server running at http://{self.host}:{self.port}")
                print(f"API requests proxied to backend at http://{self.host}:{backend_port}")
                self.frontend_process = httpd
                httpd.serve_forever()
        
        frontend_thread = threading.Thread(target=run_static_server)
        frontend_thread.daemon = True
        frontend_thread.start()
        
        time.sleep(2)
    
    def _wait_for_shutdown(self):
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, shutting down...")
        self._cleanup()
        sys.exit(0)
    
    def _cleanup(self):
        if self.frontend_process:
            try:
                self.frontend_process.terminate()
                self.frontend_process.wait(timeout=5)
                print("Frontend server stopped")
            except:
                try:
                    self.frontend_process.kill()
                except:
                    pass
        
        if self.tunnel_process:
            try:
                self.tunnel_process.terminate()
                self.tunnel_process.wait(timeout=5)
                print("Tunnel stopped")
            except:
                try:
                    self.tunnel_process.kill()
                except:
                    pass
        
        if hasattr(self, '_temp_data_file'):
            try:
                if os.path.exists(self._temp_data_file):
                    os.unlink(self._temp_data_file)
            except:
                pass
        
        print("Dashboard stopped")


def create_web_dashboard(results: Dict[str, Dict[str, ScorerResult]], 
                        trajectories: Optional[Dict] = None,
                        port: int = 5000,
                        host: str = "127.0.0.1",
                        open_browser: bool = True,
                        enable_tunnel: bool = True,
                        tunnel_token: Optional[str] = None) -> WebDashboard:
    dashboard = WebDashboard(port=port, host=host, enable_tunnel=enable_tunnel, tunnel_token=tunnel_token)
    dashboard.load_results(results, trajectories)
    dashboard.start_server(open_browser=open_browser, production=False)
    return dashboard 