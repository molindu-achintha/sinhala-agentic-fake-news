/**
 * script.js - Frontend JavaScript for Sinhala Fake News Detector
 * 
 * This script handles:
 * 1. User input and button clicks
 * 2. API calls to the backend
 * 3. Displaying results to the user
 */

// Backend API base URL
const API_BASE = 'http://localhost:8000';

// Status display element reference
let statusDiv = null;

/**
 * Show a status message to the user.
 * Creates a div if it does not exist.
 * 
 * @param {string} message - Message to display
 * @param {boolean} isError - True for error styling
 */
function showStatus(message, isError = false) {
    console.log('[showStatus]', message);

    if (!statusDiv) {
        statusDiv = document.createElement('div');
        statusDiv.id = 'statusMessage';
        statusDiv.style.cssText = 'padding: 10px; margin: 10px 0; border-radius: 5px; text-align: center;';
        document.querySelector('.input-section').appendChild(statusDiv);
    }

    statusDiv.textContent = message;
    statusDiv.style.background = isError ? '#ffdddd' : '#ddffdd';
    statusDiv.style.display = 'block';
}

/**
 * Hide the status message.
 */
function hideStatus() {
    if (statusDiv) {
        statusDiv.style.display = 'none';
    }
}

/**
 * Main check button click handler.
 * Calls the backend API to verify the claim.
 */
document.getElementById('checkBtn').addEventListener('click', async () => {
    console.log('[checkBtn] Button clicked');

    const text = document.getElementById('newsInput').value;
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');

    // Validate input
    if (!text.trim()) {
        alert("Please enter news text to verify.");
        return;
    }

    console.log('[checkBtn] Text length:', text.length);

    // Show loading state
    loader.classList.remove('hidden');
    resultSection.classList.add('hidden');
    hideStatus();

    try {
        // Step 1: Refresh latest news from sources
        showStatus('Fetching latest news...');
        console.log('[checkBtn] Calling news refresh endpoint');

        const refreshResponse = await fetch(`${API_BASE}/v1/news/refresh`);
        const refreshData = await refreshResponse.json();
        console.log('[checkBtn] News refresh response:', refreshData);

        // Step 2: Show progress
        showStatus(refreshData.message || 'News refreshed - Verifying claim...');

        // Step 3: Call predict endpoint
        console.log('[checkBtn] Calling predict endpoint');

        const response = await fetch(`${API_BASE}/v1/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                top_k: 5,
                use_pinecone: true
            })
        });

        // Check for errors
        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        console.log('[checkBtn] Predict response:', data);

        hideStatus();
        displayResult(data);

    } catch (error) {
        console.error('[checkBtn] Error:', error);
        showStatus("Error: " + error.message, true);
    } finally {
        loader.classList.add('hidden');
    }
});

/**
 * Display the prediction result to the user.
 * 
 * @param {Object} data - API response data
 */
function displayResult(data) {
    console.log('[displayResult] Displaying result');

    const resultSection = document.getElementById('resultSection');
    const claimText = document.getElementById('claimText');
    const explanationText = document.getElementById('explanationText');
    const verdictBadge = document.getElementById('verdictBadge');
    const sourcesList = document.getElementById('sourcesList');
    const reasoningList = document.getElementById('reasoningList');

    // Show result section
    resultSection.classList.remove('hidden');

    // Display Claim
    claimText.textContent = data.claim.claim_text;

    // Display Explanation (Sinhala)
    explanationText.textContent = data.verdict.explanation_si;

    // Display Verdict Badge
    verdictBadge.textContent = data.verdict.label.toUpperCase().replace(/_/g, " ");
    verdictBadge.className = `badge ${data.verdict.label}`;

    // Display Sources
    sourcesList.innerHTML = '';

    if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
        console.log('[displayResult] Evidence count:', data.retrieved_evidence.length);

        data.retrieved_evidence.forEach(ev => {
            const li = document.createElement('li');
            const source = ev.source || 'Dataset';
            const url = ev.url || '#';
            const text = ev.title || ev.text?.substring(0, 100) || 'Evidence';

            if (ev.type === 'live_news' && ev.url) {
                li.innerHTML = `<a href="${url}" target="_blank">[Live] ${source}: ${text}</a>`;
            } else {
                li.textContent = `[Dataset] ${source}: ${text}`;
            }
            sourcesList.appendChild(li);
        });
    } else if (data.verdict.citations && data.verdict.citations.length > 0) {
        data.verdict.citations.forEach(cit => {
            const li = document.createElement('li');
            li.textContent = cit;
            sourcesList.appendChild(li);
        });
    } else {
        sourcesList.innerHTML = '<li>No sources found.</li>';
    }

    // Display Reasoning Steps
    reasoningList.innerHTML = '';

    if (data.reasoning && data.reasoning.statments) {
        console.log('[displayResult] Reasoning steps:', data.reasoning.statments.length);

        data.reasoning.statments.forEach(stmt => {
            const li = document.createElement('li');
            li.textContent = `${stmt.step}: ${stmt.result}`;
            reasoningList.appendChild(li);
        });
    }
}

/**
 * Add a manual refresh button for news.
 * This allows users to refresh news without making a prediction.
 */
function addRefreshNewsButton() {
    const controls = document.querySelector('.controls');
    const refreshBtn = document.createElement('button');

    refreshBtn.id = 'refreshNewsBtn';
    refreshBtn.textContent = 'Refresh News';
    refreshBtn.style.cssText = 'margin-left: 10px; background: #4CAF50; color: white;';

    refreshBtn.onclick = async () => {
        console.log('[refreshBtn] Refresh button clicked');

        refreshBtn.disabled = true;
        refreshBtn.textContent = 'Refreshing...';

        try {
            const resp = await fetch(`${API_BASE}/v1/news/refresh`);
            const data = await resp.json();
            console.log('[refreshBtn] Response:', data);
            alert(data.message || 'News refreshed successfully');
        } catch (e) {
            console.error('[refreshBtn] Error:', e);
            alert('News refresh failed: ' + e.message);
        }

        refreshBtn.disabled = false;
        refreshBtn.textContent = 'Refresh News';
    };

    controls.appendChild(refreshBtn);
}

// Initialize when page loads
document.addEventListener('DOMContentLoaded', () => {
    console.log('[DOMContentLoaded] Page loaded, initializing...');
    addRefreshNewsButton();
    console.log('[DOMContentLoaded] Initialization complete');
});
