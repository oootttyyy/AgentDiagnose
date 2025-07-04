import { sveltekit } from '@sveltejs/kit/vite';
import { defineConfig } from 'vite';

export default defineConfig(({ command }) => {
	// Backend runs on port 9080 when frontend is on 8080 (port + 1000)
	// This matches the Python logic in web_dashboard_refactored.py
	const backendPort = 9080;
	
	// For development, we proxy to the backend
	// For production, the static files will be served by Python's HTTP server
	const shouldProxy = command === 'serve';
	
	return {
		plugins: [sveltekit()],
		server: {
			allowedHosts: true,
			...(shouldProxy && {
				proxy: {
					// Proxy API requests to the FastAPI backend
					'/api': {
						target: `http://localhost:${backendPort}`,
						changeOrigin: true,
						secure: false,
						logLevel: 'debug'
					}
				}
			})
		},
		build: {
			target: 'es2020'
		}
	};
}); 