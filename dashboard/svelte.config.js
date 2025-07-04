import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://kit.svelte.dev/docs/integrations#preprocessors
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// Use static adapter for GitHub Pages
		adapter: adapter({
			pages: 'build',
			assets: 'build',
			fallback: undefined, // GitHub Pages doesn't support SPA fallbacks
			precompress: false,
			strict: true
		}),
		// Configure paths for dedicated repository GitHub Pages deployment
		paths: {
			base: process.env.NODE_ENV === 'production' ? (process.env.BASE_PATH || '') : '',
		}
	}
};

export default config; 