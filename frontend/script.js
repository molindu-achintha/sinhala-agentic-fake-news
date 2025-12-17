/**
 * script.js
 * 
 * Frontend JavaScript for Sinhala Fake News Detector.
 * Handles user input, API calls, and result display.
 */

// Backend API URL - Change this for production
// const API_BASE = window.API_BASE || 'https://sinhala-agentic-fake-news.onrender.com';
const API_BASE = "http://localhost:8000";

// Status display element
let statusDiv = null;

/**
 * Show a status message to the user.
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
    statusDiv.style.display = 'block';

    if (isError) {
        statusDiv.style.backgroundColor = '#ffebee';
        statusDiv.style.color = '#c62828';
    } else {
        statusDiv.style.backgroundColor = '#e3f2fd';
        statusDiv.style.color = '#1565c0';
    }
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
 * Check claim using the API.
 */
async function checkClaim() {
    console.log('[checkClaim] Starting verification');

    const claimInput = document.getElementById('claimInput');
    const claim = claimInput.value.trim();

    if (!claim) {
        showStatus('Please enter a claim to verify', true);
        return;
    }

    showStatus('Verifying claim...');

    try {
        const response = await fetch(API_BASE + '/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ text: claim })
        });

        if (!response.ok) {
            throw new Error('API request failed: ' + response.status);
        }

        const result = await response.json();
        console.log('[checkClaim] Result:', result);

        hideStatus();
        displayResult(result);

    } catch (error) {
        console.error('[checkClaim] Error:', error);
        showStatus('Error: ' + error.message, true);
    }
}

/**
 * Display verification result.
 */
function displayResult(result) {
    console.log('[displayResult] Showing result');

    const resultSection = document.getElementById('resultSection');
    if (!resultSection) {
        console.error('[displayResult] Result section not found');
        return;
    }

    // Get verdict info
    const verdict = result.verdict || {};
    const label = verdict.label || 'unknown';
    const confidence = verdict.confidence || 0;
    const explanationSi = verdict.explanation_si || '';
    const explanationEn = verdict.explanation_en || '';

    // Get evidence info
    const evidence = result.evidence || {};
    const labeledCount = evidence.labeled_count || 0;
    const topSimilarity = evidence.top_similarity || 0;

    // Get cache info
    const fromCache = result.from_cache || false;

    // Build result HTML
    let html = '<div class="result-card">';

    // Verdict badge
    html += '<div class="verdict-badge verdict-' + label + '">';
    html += getVerdictText(label);
    html += '</div>';

    // Confidence
    html += '<div class="confidence">';
    html += 'Confidence: ' + Math.round(confidence * 100) + '%';
    html += '</div>';

    // Cache indicator
    if (fromCache) {
        html += '<div class="cache-indicator">From cache</div>';
    }

    // Explanations
    html += '<div class="explanations">';
    if (explanationSi) {
        html += '<p class="sinhala">' + explanationSi + '</p>';
    }
    if (explanationEn) {
        html += '<p class="english">' + explanationEn + '</p>';
    }
    html += '</div>';

    // Evidence summary
    html += '<div class="evidence-summary">';
    html += '<p>Evidence found: ' + labeledCount + ' labeled documents</p>';
    html += '<p>Top similarity: ' + Math.round(topSimilarity * 100) + '%</p>';
    html += '</div>';

    // Citations
    const citations = evidence.citations || [];
    if (citations.length > 0) {
        html += '<div class="citations">';
        html += '<h4>Sources</h4>';
        html += '<ul>';
        for (let cite of citations) {
            html += '<li>[' + cite.source + '] ' + cite.text + '</li>';
        }
        html += '</ul>';
        html += '</div>';
    }

    html += '</div>';

    resultSection.innerHTML = html;
    resultSection.style.display = 'block';
}

/**
 * Get verdict display text.
 */
function getVerdictText(label) {
    const texts = {
        'true': 'TRUE / සත්‍ය',
        'false': 'FALSE / අසත්‍ය',
        'misleading': 'MISLEADING / නොමඟයවන',
        'needs_verification': 'NEEDS VERIFICATION / තහවුරු කළ යුතුය',
        'likely_true': 'LIKELY TRUE / බොහෝ දුරට සත්‍ය',
        'likely_false': 'LIKELY FALSE / බොහෝ දුරට අසත්‍ය',
        'unverified': 'UNVERIFIED / තහවුරු නොවූ'
    };
    return texts[label] || label.toUpperCase();
}

/**
 * Check API health.
 */
async function checkHealth() {
    console.log('[checkHealth] Checking API health');

    try {
        const response = await fetch(API_BASE + '/v1/health');
        const data = await response.json();
        console.log('[checkHealth] API is healthy:', data);
        return true;
    } catch (error) {
        console.error('[checkHealth] API is not available:', error);
        return false;
    }
}

/**
 * Initialize the application.
 */
function init() {
    console.log('[init] Initializing Sinhala Fake News Detector');
    console.log('[init] API URL:', API_BASE);

    // Check API health
    checkHealth().then(healthy => {
        if (!healthy) {
            showStatus('Warning: Backend API is not available', true);
        }
    });

    // Set up event listeners
    const checkButton = document.getElementById('checkButton');
    if (checkButton) {
        checkButton.addEventListener('click', checkClaim);
    }

    const claimInput = document.getElementById('claimInput');
    if (claimInput) {
        claimInput.addEventListener('keypress', function (e) {
            if (e.key === 'Enter') {
                checkClaim();
            }
        });
    }

    console.log('[init] Ready');
}

// Start when DOM is ready
document.addEventListener('DOMContentLoaded', init);
