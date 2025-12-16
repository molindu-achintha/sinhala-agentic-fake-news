const API_BASE = 'http://localhost:8000';

// Status display element
let statusDiv = null;

function showStatus(message, isError = false) {
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

function hideStatus() {
    if (statusDiv) statusDiv.style.display = 'none';
}

document.getElementById('checkBtn').addEventListener('click', async () => {
    const text = document.getElementById('newsInput').value;
    const loader = document.getElementById('loader');
    const resultSection = document.getElementById('resultSection');

    if (!text.trim()) {
        alert("කරුණාකර පුවතක් ඇතුළත් කරන්න.");
        return;
    }

    loader.classList.remove('hidden');
    resultSection.classList.add('hidden');
    hideStatus();

    try {
        // Step 1: Refresh latest news from sources
        showStatus('🔄 නවතම පුවත් ලබා ගනිමින්... (Fetching latest news)');
        const refreshResponse = await fetch(`${API_BASE}/v1/news/refresh`);
        const refreshData = await refreshResponse.json();
        console.log('News refreshed:', refreshData);

        // Step 2: Show how many articles found
        showStatus(`📰 ${refreshData.message || 'News refreshed'} - සත්‍යාපනය කරමින්...`);

        // Step 3: Call predict endpoint with live news search enabled
        const response = await fetch(`${API_BASE}/v1/predict`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                top_k: 5,
                use_live_news: true  // Flag to use scraped news
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        hideStatus();
        displayResult(data);

    } catch (error) {
        console.error(error);
        showStatus("දෝෂයක් සිදුවිය: " + error.message, true);
    } finally {
        loader.classList.add('hidden');
    }
});

function displayResult(data) {
    const resultSection = document.getElementById('resultSection');
    const claimText = document.getElementById('claimText');
    const explanationText = document.getElementById('explanationText');
    const verdictBadge = document.getElementById('verdictBadge');
    const sourcesList = document.getElementById('sourcesList');
    const reasoningList = document.getElementById('reasoningList');

    resultSection.classList.remove('hidden');

    // Display Claim
    claimText.textContent = data.claim.claim_text;

    // Display Explanation
    explanationText.textContent = data.verdict.explanation_si;

    // Display Verdict
    verdictBadge.textContent = data.verdict.label.toUpperCase().replace(/_/g, " ");
    verdictBadge.className = `badge ${data.verdict.label}`;

    // Display Sources (including live news)
    sourcesList.innerHTML = '';

    // Show retrieved evidence sources
    if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
        data.retrieved_evidence.forEach(ev => {
            const li = document.createElement('li');
            const source = ev.source || 'Dataset';
            const url = ev.url || '#';
            const text = ev.title || ev.text?.substring(0, 100) || 'Evidence';

            if (ev.type === 'live_news' && ev.url) {
                li.innerHTML = `<a href="${url}" target="_blank">📰 ${source}: ${text}</a>`;
            } else {
                li.textContent = `📄 ${source}: ${text}`;
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
        sourcesList.innerHTML = '<li>මූලාශ්‍ර හමු නොවීය.</li>';
    }

    // Display Reasoning
    reasoningList.innerHTML = '';
    if (data.reasoning && data.reasoning.statments) {
        data.reasoning.statments.forEach(stmt => {
            const li = document.createElement('li');
            li.textContent = `${stmt.step}: ${stmt.result}`;
            reasoningList.appendChild(li);
        });
    }
}

// Optional: Add button to manually refresh news
function addRefreshNewsButton() {
    const controls = document.querySelector('.controls');
    const refreshBtn = document.createElement('button');
    refreshBtn.id = 'refreshNewsBtn';
    refreshBtn.textContent = '🔄 පුවත් යාවත්කාලීන කරන්න';
    refreshBtn.style.cssText = 'margin-left: 10px; background: #4CAF50; color: white;';
    refreshBtn.onclick = async () => {
        refreshBtn.disabled = true;
        refreshBtn.textContent = 'යාවත්කාලීන කරමින්...';
        try {
            const resp = await fetch(`${API_BASE}/v1/news/refresh`);
            const data = await resp.json();
            alert(`✅ ${data.message}`);
        } catch (e) {
            alert('News refresh failed: ' + e.message);
        }
        refreshBtn.disabled = false;
        refreshBtn.textContent = '🔄 පුවත් යාවත්කාලීන කරන්න';
    };
    controls.appendChild(refreshBtn);
}

// Add refresh button when page loads
document.addEventListener('DOMContentLoaded', addRefreshNewsButton);
