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

    // Get selected LLM provider
    const llmProviderSelect = document.getElementById('llmProvider');
    const llmProvider = llmProviderSelect ? llmProviderSelect.value : 'groq';
    console.log('[checkClaim] Using LLM provider:', llmProvider);

    try {
        const response = await fetch(API_BASE + '/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: claim,
                llm_provider: llmProvider
            })
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
    const explanationEn = verdict.explanation_en || verdict.detailed_explanation || '';
    const llmPowered = verdict.llm_powered || false;

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
    if (llmPowered) {
        html += ' <span style="background: #10b981; padding: 3px 8px; border-radius: 10px; font-size: 0.8rem; margin-left: 10px;">🤖 AI Verified</span>';
    }
    html += '</div>';

    // Cache indicator
    if (fromCache) {
        html += '<div class="cache-indicator">From cache</div>';
    }

    // Sinhala explanation (brief)
    if (explanationSi) {
        html += '<div style="margin: 15px 0; padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px;">';
        html += '<p class="sinhala" style="font-size: 1.1rem;">' + explanationSi + '</p>';
        html += '</div>';
    }

    // Main LLM Reasoning Container (with streaming)
    html += '<div id="aiReasoningContainer" style="margin-top: 20px; padding: 25px; background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(20,20,40,0.6) 100%); border-radius: 20px; text-align: left; border: 1px solid rgba(0,212,255,0.3);">';
    html += '<div id="aiReasoningText" style="white-space: pre-wrap; line-height: 1.8; font-size: 1rem; color: #e0e0e0;"></div>';
    html += '</div>';
    // Evidence summary (collapsed)
    html += '<details style="margin-top: 15px; color: #888;">';
    html += '<summary style="cursor: pointer;">Evidence Details (' + labeledCount + ' documents, ' + Math.round(topSimilarity * 100) + '% similarity)</summary>';
    html += '<div style="padding: 10px;">';

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

    html += '</div></details>';
    html += '</div>';

    resultSection.innerHTML = html;
    resultSection.style.display = 'block';

    // Start typewriter effect for AI reasoning (the main explanation from LLM)
    if (explanationEn && explanationEn.length > 50) {
        typewriterEffect('aiReasoningText', explanationEn, 10);
    } else {
        const aiReasoningText = document.getElementById('aiReasoningText');
        if (aiReasoningText) {
            aiReasoningText.innerHTML = explanationEn || '<span style="color: #888;">Generating AI analysis... (Check API keys in .env if this persists)</span>';
        }
    }
}

/**
 * Typewriter effect for streaming text display.
 */
function typewriterEffect(elementId, text, speed) {
    const element = document.getElementById(elementId);
    if (!element) return;

    let index = 0;
    element.innerHTML = '<span class="cursor" style="animation: blink 1s infinite;">▌</span>';

    // Add cursor blink animation
    const style = document.createElement('style');
    style.textContent = '@keyframes blink { 0%, 50% { opacity: 1; } 51%, 100% { opacity: 0; } }';
    document.head.appendChild(style);

    function type() {
        if (index < text.length) {
            // Get next chunk (for faster typing)
            const chunk = text.slice(index, index + 3);
            element.innerHTML = formatMarkdown(text.slice(0, index + chunk.length)) + '<span class="cursor" style="animation: blink 1s infinite;">▌</span>';
            index += chunk.length;
            setTimeout(type, speed);
        } else {
            // Remove cursor when done
            element.innerHTML = formatMarkdown(text);
        }
    }

    type();
}

/**
 * Basic markdown to HTML formatting.
 */
function formatMarkdown(text) {
    return text
        .replace(/\*\*(.*?)\*\*/g, '<strong style="color: #00d4ff;">$1</strong>')
        .replace(/\n/g, '<br>')
        .replace(/✔/g, '<span style="color: #10b981;">✔</span>')
        .replace(/❌/g, '<span style="color: #ef4444;">❌</span>')
        .replace(/✅/g, '<span style="color: #10b981;">✅</span>')
        .replace(/📌/g, '<span style="font-size: 1.2em;">📌</span>')
        .replace(/🧾/g, '<span style="font-size: 1.2em;">🧾</span>')
        .replace(/🔎/g, '<span style="font-size: 1.2em;">🔎</span>');
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
