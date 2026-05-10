document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('themeToggle');
    const newsContainer = document.getElementById('newsContainer');
    const refreshBtn = document.getElementById('refreshNews');
    
    // ==========================================
    // Theme Management
    // ==========================================
    if (localStorage.getItem('theme') === 'dark') {
        document.body.classList.add('dark-theme');
    } else {
        document.body.classList.remove('dark-theme');
    }

    themeToggle.addEventListener('click', () => {
        const isDark = document.body.classList.toggle('dark-theme');
        localStorage.setItem('theme', isDark ? 'dark' : 'light');
    });

    // ==========================================
    // News Fetching & Display
    // ==========================================
    function showSkeleton() {
        newsContainer.innerHTML = `
            <div class="skeleton-loader">
                <div class="skeleton-line title"></div>
                <div class="skeleton-line text"></div>
                <div class="skeleton-line text short"></div>
            </div>
            <div class="skeleton-loader" style="margin-top: 2rem">
                <div class="skeleton-line title"></div>
                <div class="skeleton-line text"></div>
                <div class="skeleton-line text short"></div>
            </div>
        `;
    }

    function formatTime(dateStr) {
        const date = new Date(dateStr);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000 / 60); // minutes
        
        if (diff < 60) return `${diff}m ago`;
        if (diff < 1440) return `${Math.floor(diff/60)}h ago`;
        return date.toLocaleDateString();
    }

    async function loadNews() {
        showSkeleton();
        if (refreshBtn) refreshBtn.style.opacity = '0.5';
        
        try {
            const response = await fetch('/api/news');
            if (!response.ok) throw new Error('Failed to fetch news');
            
            const data = await response.json();
            displayNews(data.articles);
        } catch (error) {
            console.error('News Error:', error);
            newsContainer.innerHTML = `
                <div class="error-msg" style="padding: 2rem; text-align: center; color: var(--text-secondary);">
                    <p>Unable to load live feed. Please try again later.</p>
                </div>
            `;
        } finally {
            if (refreshBtn) refreshBtn.style.opacity = '1';
        }
    }

    function displayNews(articles) {
        if (!articles || articles.length === 0) {
            newsContainer.innerHTML = '<p style="text-align: center; color: var(--text-secondary);">No news available at the moment.</p>';
            return;
        }

        newsContainer.innerHTML = articles.map(article => `
            <div class="news-item" style="margin-bottom: 1.5rem; border-bottom: 1px solid var(--border-color); padding-bottom: 1rem;">
                <h4 style="font-family: var(--font-serif); font-size: 1.1rem; margin-bottom: 0.5rem;">
                    <a href="${article.url}" target="_blank" style="color: var(--text-primary);">${article.title}</a>
                </h4>
                <div style="display: flex; gap: 12px; font-size: 0.8rem; color: var(--text-muted);">
                    <span>${article.source.name}</span>
                    <span>•</span>
                    <span>${formatTime(article.publishedAt)}</span>
                </div>
            </div>
        `).join('');
    }

    // Initial Load
    loadNews();

    // Refresh Event
    if (refreshBtn) {
        refreshBtn.addEventListener('click', loadNews);
    }
});