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

    try {
        const response = await fetch('http://localhost:8080/v1/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                text: text,
                top_k: 3
            })
        });

        if (!response.ok) {
            throw new Error(`API Error: ${response.status}`);
        }

        const data = await response.json();
        displayResult(data);

    } catch (error) {
        console.error(error);
        alert("දෝෂයක් සිදුවිය: " + error.message);
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

    // Display Sources
    sourcesList.innerHTML = '';
    if (data.verdict.citations && data.verdict.citations.length > 0) {
        data.verdict.citations.forEach(cit => {
            const li = document.createElement('li');
            li.textContent = cit;
            sourcesList.appendChild(li);
        });
    } else {
        sourcesList.innerHTML = '<li>මූලාශ්‍ර හමු නොවීය.</li>';
    }

    // Display Reasoning (if available)
    reasoningList.innerHTML = '';
    if (data.reasoning && data.reasoning.statments) {
        data.reasoning.statments.forEach(stmt => {
            const li = document.createElement('li');
            li.textContent = `${stmt.step}: ${stmt.result}`;
            reasoningList.appendChild(li);
        });
    }
}
