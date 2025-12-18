/**
 * script.js
 * 
 * Frontend JavaScript for Sinhala Fake News Detector.
 * Handles user input, API calls, and result display.
 */

// Backend API URL - Change this for production
// const API_BASE = window.API_BASE || 'https://sinhala-agentic-fake-news.onrender.com';
const API_BASE = "https://sinhala-agentic-fake-news.onrender.com";

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

    // Clear previous results immediately
    const resultSection = document.getElementById('resultSection');
    if (resultSection) {
        resultSection.innerHTML = '';
        resultSection.style.display = 'none';
    }

    // Show initial status with animation
    showVerificationProgress('🔍 Starting verification...');

    // Get selected LLM provider
    const llmProviderSelect = document.getElementById('llmProvider');
    const llmProvider = llmProviderSelect ? llmProviderSelect.value : 'deepresearch';

    // Get Vector DB toggle
    const useVectorDbInput = document.getElementById('useVectorDb');
    const useVectorDb = useVectorDbInput ? useVectorDbInput.checked : true;

    console.log('[checkClaim] Using LLM provider:', llmProvider);
    console.log('[checkClaim] Use Vector DB:', useVectorDb);

    // Update status with steps
    updateProgressStep('📝 Step 1: Analyzing claim...');
    await sleep(500);
    updateProgressStep('🌐 Step 2: Research Agent gathering evidence...');

    try {
        const response = await fetch(API_BASE + '/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: claim,
                llm_provider: llmProvider,
                use_vector_db: useVectorDb
            })
        });

        if (!response.ok) {
            throw new Error('API request failed: ' + response.status);
        }

        updateProgressStep('⚖️ Step 3: Judge Agent analyzing evidence...');
        const result = await response.json();
        console.log('[checkClaim] Result:', result);

        updateProgressStep('✅ Verification complete!');
        await sleep(300);
        hideVerificationProgress();
        displayResult(result);

    } catch (error) {
        console.error('[checkClaim] Error:', error);
        hideVerificationProgress();
        showStatus('Error: ' + error.message, true);
    }
}

/**
 * Show verification progress panel.
 */
function showVerificationProgress(message) {
    let progressDiv = document.getElementById('verificationProgress');
    if (!progressDiv) {
        progressDiv = document.createElement('div');
        progressDiv.id = 'verificationProgress';
        progressDiv.style.cssText = 'padding: 20px; margin: 15px 0; border-radius: 15px; background: linear-gradient(135deg, rgba(0,212,255,0.1) 0%, rgba(123,44,191,0.1) 100%); border: 1px solid rgba(0,212,255,0.3); text-align: center;';
        document.querySelector('.input-section').appendChild(progressDiv);
    }
    progressDiv.innerHTML = `
        <div style="font-size: 1.2rem; margin-bottom: 10px;">${message}</div>
        <div id="progressSteps" style="color: #888; font-size: 0.9rem;"></div>
        <div style="margin-top: 15px;">
            <div class="spinner" style="display: inline-block; width: 20px; height: 20px; border: 3px solid rgba(0,212,255,0.3); border-top: 3px solid #00d4ff; border-radius: 50%; animation: spin 1s linear infinite;"></div>
        </div>
        <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    `;
    progressDiv.style.display = 'block';
}

/**
 * Update progress step.
 */
function updateProgressStep(step) {
    const stepsDiv = document.getElementById('progressSteps');
    if (stepsDiv) {
        stepsDiv.innerHTML += `<div style="margin: 5px 0;">${step}</div>`;
    }
}

/**
 * Hide verification progress.
 */
function hideVerificationProgress() {
    const progressDiv = document.getElementById('verificationProgress');
    if (progressDiv) {
        progressDiv.style.display = 'none';
    }
}

/**
 * Sleep helper.
 */
function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
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
    const explanationSi = verdict.explanation_si || verdict.detailed_explanation || '';
    const llmPowered = verdict.llm_powered || false;
    const citations = verdict.citations || [];

    // Get claim info
    const claim = result.claim || {};
    const originalClaim = claim.original || '';
    const normalizedSi = claim.normalized_si || '';
    const normalizedEn = claim.normalized_en || '';

    // Get evidence info
    const evidence = result.evidence || {};
    const webCount = evidence.web_count || 0;
    const researchEvidence = result.research_evidence || {};
    const evidenceList = researchEvidence.evidence || [];

    // Get cache info
    const fromCache = result.from_cache || false;

    // Build result HTML with structured sections
    let html = '<div class="result-card">';

    // ========== VERDICT HEADER ==========
    html += '<div style="text-align: center; margin-bottom: 25px;">';
    html += '<div class="verdict-badge verdict-' + label + '" style="font-size: 1.8rem; padding: 20px 50px;">';
    html += getVerdictText(label);
    html += '</div>';
    html += '<div class="confidence" style="margin-top: 15px; font-size: 1.3rem;">';
    html += 'Confidence: ' + Math.round(confidence * 100) + '%';
    if (llmPowered) {
        html += ' <span style="background: #10b981; padding: 5px 12px; border-radius: 12px; font-size: 0.9rem; margin-left: 10px;">🤖 AI Verified</span>';
    }
    if (fromCache) {
        html += ' <span style="background: #6366f1; padding: 5px 12px; border-radius: 12px; font-size: 0.9rem; margin-left: 10px;">📦 Cached</span>';
    }
    html += '</div>';
    html += '</div>';

    // ========== CLAIM SECTION ==========
    if (normalizedSi || normalizedEn) {
        html += '<div style="margin: 20px 0; padding: 20px; background: rgba(0,0,0,0.2); border-radius: 15px; border-left: 4px solid #00d4ff;">';
        html += '<h3 style="color: #00d4ff; margin-bottom: 10px; font-size: 0.9rem; text-transform: uppercase;">📋 Analyzed Claim</h3>';
        if (normalizedSi) {
            html += '<p style="font-size: 1.1rem; margin-bottom: 8px;">' + normalizedSi + '</p>';
        }
        if (normalizedEn && normalizedEn !== normalizedSi) {
            html += '<p style="color: #888; font-size: 0.95rem; font-style: italic;">' + normalizedEn + '</p>';
        }
        html += '</div>';
    }

    // ========== MAIN EXPLANATION (SINHALA) ==========
    html += '<div style="margin: 20px 0;">';
    html += '<h3 style="color: #00d4ff; margin-bottom: 15px; font-size: 0.9rem; text-transform: uppercase;">📝 Detailed Analysis</h3>';
    html += '<div id="aiReasoningContainer" style="padding: 25px; background: linear-gradient(135deg, rgba(0,0,0,0.4) 0%, rgba(20,20,40,0.6) 100%); border-radius: 20px; border: 1px solid rgba(0,212,255,0.2);">';
    html += '<div id="aiReasoningText" style="white-space: pre-wrap; line-height: 1.9; font-size: 1.05rem; color: #e0e0e0;"></div>';
    html += '</div>';
    html += '</div>';

    // ========== EVIDENCE SECTION ==========
    if (evidenceList.length > 0) {
        html += '<div style="margin: 20px 0;">';
        html += '<h3 style="color: #00d4ff; margin-bottom: 15px; font-size: 0.9rem; text-transform: uppercase;">🔍 Evidence Sources (' + evidenceList.length + ' found)</h3>';
        html += '<div style="display: grid; gap: 10px;">';

        for (let ev of evidenceList) {
            const relationColor = ev.relation === 'SUPPORTS' ? '#10b981' : (ev.relation === 'REFUTES' ? '#ef4444' : '#888');
            const relationIcon = ev.relation === 'SUPPORTS' ? '✓' : (ev.relation === 'REFUTES' ? '✗' : '?');

            html += '<div style="padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; border-left: 3px solid ' + relationColor + ';">';
            html += '<div style="display: flex; justify-content: space-between; margin-bottom: 8px;">';
            html += '<span style="font-weight: bold; color: #fff;">[' + ev.id + '] ' + (ev.outlet || 'Unknown Source') + '</span>';
            html += '<span style="color: ' + relationColor + '; font-weight: bold;">' + relationIcon + ' ' + ev.relation + '</span>';
            html += '</div>';
            if (ev.snippet) {
                html += '<p style="color: #aaa; font-size: 0.9rem; margin: 8px 0;">"' + ev.snippet.substring(0, 200) + (ev.snippet.length > 200 ? '...' : '') + '"</p>';
            }
            if (ev.url) {
                html += '<a href="' + ev.url + '" target="_blank" style="color: #00d4ff; font-size: 0.8rem; text-decoration: none;">🔗 View Source</a>';
            }
            html += '</div>';
        }

        html += '</div>';
        html += '</div>';
    }

    // ========== CITATIONS FOOTER ==========
    if (citations.length > 0) {
        html += '<details style="margin-top: 20px; color: #888;">';
        html += '<summary style="cursor: pointer; padding: 10px;">📚 Citation References (' + citations.length + ')</summary>';
        html += '<div style="padding: 15px; background: rgba(0,0,0,0.2); border-radius: 10px; margin-top: 10px;">';
        for (let cite of citations) {
            html += '<div style="margin: 5px 0; font-size: 0.85rem;">[' + cite.id + '] ' + cite.outlet + ' - <a href="' + cite.url + '" target="_blank" style="color: #00d4ff;">Link</a></div>';
        }
        html += '</div>';
        html += '</details>';
    }

    html += '</div>';

    resultSection.innerHTML = html;
    resultSection.style.display = 'block';

    // Start typewriter effect for AI reasoning
    const aiReasoningText = document.getElementById('aiReasoningText');
    if (aiReasoningText) {
        if (explanationSi && explanationSi.length > 50) {
            typewriterEffect('aiReasoningText', explanationSi, 8);
        } else if (explanationSi) {
            aiReasoningText.innerHTML = explanationSi;
        } else {
            aiReasoningText.innerHTML = '<span style="color: #888;">Analysis could not be generated. Check API keys in .env file.</span>';
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
