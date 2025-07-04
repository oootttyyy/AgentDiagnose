<script>
	import { onMount } from 'svelte';
	import { writable } from 'svelte/store';
	import PlotComponent from '../lib/components/PlotComponent.svelte';
	import TrajectoryView from '../lib/components/TrajectoryView.svelte';
	import MetricsGrid from '../lib/components/MetricsGrid.svelte';
	import EmbeddingVisualization from '../lib/components/EmbeddingVisualization.svelte';
	import NavigationPathView from '../lib/components/NavigationPathView.svelte';
	import ReasoningTagCloud from '../lib/components/ReasoningTagCloud.svelte';
	import ActionPhrasesTagCloud from '../lib/components/ActionPhrasesTagCloud.svelte';

	// Stores for reactive data
	const resultsData = writable({});
	const trajectoriesData = writable({});
	const activeTab = writable('summary');

	// Dynamic API base URL - for production use proxy, for dev use relative path
	function getApiBaseUrl() {
		// Use relative path which will be proxied to backend by Vite
		return '/api';
	}

	let API_BASE = getApiBaseUrl();

	// Dynamic tabs and scorers
	let availableScorers = [];
	let dynamicTabs = [];

	// Header behavior variables
	let headerCollapsed = false;
	let firstClickDetected = false;
	let lastScrollTop = 0;
	let headerElement;

	onMount(async () => {
		// Ensure API_BASE is set correctly in the browser
		API_BASE = getApiBaseUrl();
		
		// Load initial data
		await loadData();
		
		// Setup header behavior exactly like original
		setupHeaderBehavior();
	});

	async function loadData() {
		try {
			// Only need to fetch results - it contains both summary and trajectories
			const resultsResponse = await fetch(`${API_BASE}/results`);

			const results = await resultsResponse.json();

			resultsData.set(results);
			// No need for separate trajectories call - results already contains trajectories
			trajectoriesData.set(results.trajectories || {});
			
			// Extract available scorers and setup dynamic tabs
			if (results.summary && results.summary.scorers_used) {
				availableScorers = results.summary.scorers_used;
				setupDynamicTabs();
			}
		} catch (error) {
			console.error('Error loading data:', error);
		}
	}

	function setupDynamicTabs() {
		// Create base tabs (always present)
		dynamicTabs = [
			{ id: 'summary', label: 'SUMMARY' },
			{ id: 'view-trajectory', label: 'VIEW TRAJECTORY' },
			{ id: 'embeddings', label: 'EMBEDDINGS' },
			{ id: 'reasoning', label: 'TAG-CLOUD: REASONING' },
			{ id: 'action-phrases', label: 'TAG-CLOUD: ACTION PHRASE' }
		];
		
		// Add dynamic scorer tabs
		availableScorers.forEach(scorer => {
			// Dynamic formatting: handle underscores and add spaces before capitals
			let tabText = scorer
				.replace(/_/g, ' ')  // Replace underscores with spaces
				.replace(/([a-z])([A-Z])/g, '$1 $2')  // Add space before capitals
				.toUpperCase();
			
			dynamicTabs.push({
				id: `scorer-${scorer}`,
				label: tabText,
				scorerName: scorer
			});
		});
		
		// Trigger reactivity
		dynamicTabs = [...dynamicTabs];
	}

	function setupHeaderBehavior() {
		console.log('Setting up header behavior, header found:', headerElement);
		
		// Handle first click anywhere on the page
		function handleFirstClick(e) {
			console.log('First click detected, folding header');
			if (!firstClickDetected) {
				firstClickDetected = true;
				headerCollapsed = true;
				if (headerElement) {
					headerElement.classList.add('hidden');
				}
				console.log('Header hidden class added');
				
				// Remove this event listener after first use
				document.removeEventListener('click', handleFirstClick);
			}
		}
		
		document.addEventListener('click', handleFirstClick);
		
		// Debouncing and transition lock variables
		let scrollTimeout;
		let isTransitioning = false; // Prevent scroll handling during transitions
		let transitionCooldown = 500; // Cooldown period after transitions
		
		// Handle scroll behavior with debouncing and transition lock
		function handleScroll() {
			if (!firstClickDetected || isTransitioning) return; // Don't handle scroll during transitions
			
			clearTimeout(scrollTimeout);
			scrollTimeout = setTimeout(() => {
				const currentScrollTop = window.pageYOffset || document.documentElement.scrollTop;
				const scrollDifference = currentScrollTop - lastScrollTop;
				
				console.log('Scroll detected - Position:', currentScrollTop, 'Last:', lastScrollTop, 'Diff:', scrollDifference, 'Collapsed:', headerCollapsed);
				
				// Only act if scroll difference is significant and we're not transitioning
				if (!isTransitioning) {
					if (scrollDifference > 0 && currentScrollTop > 30) {
						// Scrolling down significantly - hide header (easier to hide)
						if (!headerCollapsed) {
							isTransitioning = true;
							headerCollapsed = true;
							if (headerElement) {
								headerElement.classList.add('hidden');
							}
							console.log('Header hidden due to scroll down');
							
							// Release transition lock after animation
							setTimeout(() => {
								isTransitioning = false;
								lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;
							}, transitionCooldown);
						}
					} else if (scrollDifference < -500 && currentScrollTop <= 10) {
						// Scrolling up near the top (within 10px) - show header (harder to show)
						if (headerCollapsed) {
							isTransitioning = true;
							headerCollapsed = false;
							if (headerElement) {
								headerElement.classList.remove('hidden');
							}
							console.log('Header shown due to scroll up near top');
							
							// Release transition lock after animation
							setTimeout(() => {
								isTransitioning = false;
								lastScrollTop = window.pageYOffset || document.documentElement.scrollTop;
							}, transitionCooldown);
						}
					}
					
					// Only update lastScrollTop if we're not transitioning
					if (!isTransitioning) {
						lastScrollTop = currentScrollTop <= 0 ? 0 : currentScrollTop;
					}
				}
			}, 100); // Increased debounce delay
		}
		
		// Attach scroll listener to window only
		window.addEventListener('scroll', handleScroll, { passive: true });
	}

	function switchTab(tabId) {
		activeTab.set(tabId);
		
		// Load content based on tab
		if (tabId === 'view-trajectory') {
			// TrajectoryView component will handle loading
		} else if (tabId === 'embeddings') {
			// EmbeddingVisualization component will handle loading
		} else if (tabId === 'reasoning') {
			// ReasoningTagCloud component will handle loading
		} else if (tabId === 'action-phrases') {
			// ActionPhrasesTagCloud component will handle loading
		} else if (tabId.startsWith('scorer-')) {
			const scorerName = tabId.replace('scorer-', '');
			// Plot will be loaded by PlotComponent
		}
	}

	// Handle navigation events from MetricsGrid
	function handleNavigate(event) {
		const { tabId } = event.detail;
		switchTab(tabId);
	}
</script>

<div class="container">
	<!-- Header -->
	<div class="header" bind:this={headerElement}>
		<h1>A Trajectory Inspector Tool</h1>
	</div>

	<!-- Tabs -->
	<div class="tabs-container">
		<div class="tabs">
			{#each dynamicTabs as tab}
				<button 
					class="tab" 
					class:active={$activeTab === tab.id}
					on:click={() => switchTab(tab.id)}
				>
					{tab.label}
				</button>
			{/each}
		</div>

		<!-- Tab Content -->
		{#if $activeTab === 'summary'}
			<div class="tab-content active">
				<MetricsGrid {resultsData} on:navigate={handleNavigate} />
			</div>
		{:else if $activeTab === 'view-trajectory'}
			<div class="tab-content active">
				<h2 style="margin-bottom: 20px; font-size: 24px; font-weight: 600;">Trajectory Viewer</h2>
				<TrajectoryView />
			</div>
		{:else if $activeTab === 'embeddings'}
			<div class="tab-content active">
				<EmbeddingVisualization apiBase={API_BASE} />
			</div>
		{:else if $activeTab === 'reasoning'}
			<div class="tab-content active">
				<h2 style="margin-bottom: 20px; font-size: 24px; font-weight: 600;">Reasoning Word Cloud</h2>
				<ReasoningTagCloud apiBase={API_BASE} />
			</div>
		{:else if $activeTab === 'action-phrases'}
			<div class="tab-content active">
				<h2 style="margin-bottom: 20px; font-size: 24px; font-weight: 600;">Action Phrases Word Cloud</h2>
				<ActionPhrasesTagCloud apiBase={API_BASE} />
			</div>
		{:else if $activeTab.startsWith('scorer-')}
			{@const scorerName = $activeTab.replace('scorer-', '')}
			{@const displayName = scorerName.replace(/_/g, ' ').replace(/([a-z])([A-Z])/g, '$1 $2')}
			<div class="tab-content active">
				<h2 style="margin-bottom: 20px; font-size: 24px; font-weight: 600;">{displayName} Analysis</h2>
				{#if scorerName === 'NavigationPath'}
					<NavigationPathView />
				{:else}
					<PlotComponent plotType="scorer/{scorerName}" title="{displayName} Distribution" />
				{/if}
			</div>
		{/if}
	</div>
</div>

<style>
	.container {
		max-width: 1200px;
		margin: 0 auto;
		padding: 40px 20px;
	}
	
	.header {
		text-align: center;
		margin-bottom: 20px;
		position: sticky;
		top: 0;
		background-color: #f5f5f5;
		z-index: 1000;
		transition: all 0.3s ease-in-out;
		padding: 10px 0;
		overflow: hidden;
	}
	
	:global(.header.hidden) {
		transform: translateY(-100%);
		height: 0;
		padding: 0;
		margin-bottom: 0;
		opacity: 0;
		visibility: hidden;
		pointer-events: none;
	}
	
	.header h1 {
		font-size: 48px;
		font-weight: 800;
		letter-spacing: 1px;
		color: #2c2c2c;
		margin-bottom: 15px;
	}
	
	.tabs-container {
		margin-bottom: 40px;
	}
	
	.tabs {
		display: flex;
		border-bottom: 3px solid #333;
		margin-bottom: 40px;
		flex-wrap: wrap;
	}
	
	.tab {
		background: none;
		border: none;
		padding: 15px 30px;
		font-size: 18px;
		font-weight: 600;
		color: #666;
		cursor: pointer;
		transition: all 0.3s ease;
		border-bottom: 3px solid transparent;
		margin-bottom: -3px;
	}
	
	.tab.active {
		color: #333;
		border-bottom: 3px solid #333;
	}
	
	.tab:hover {
		color: #333;
	}
	
	.tab-content {
		display: none;
		min-height: 100vh;
	}
	
	.tab-content.active {
		display: block;
	}

	@media (max-width: 768px) {
		.header h1 {
			font-size: 32px;
		}
		
		.tab {
			font-size: 16px;
			padding: 12px 20px;
		}
	}
</style> 