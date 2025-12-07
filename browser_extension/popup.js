document.getElementById('check').addEventListener('click', async () => {
    const text = document.getElementById('text').value;
    const output = document.getElementById('output');

    if (!text) return;

    output.textContent = "Checking...";

    try {
        const res = await fetch('http://localhost:8080/v1/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ text: text })
        });
        const data = await res.json();
        output.innerHTML = `<strong>Verdict:</strong> ${data.verdict.label}<br/>${data.verdict.explanation_si}`;
    } catch (e) {
        output.textContent = "Error connecting to backend.";
    }
});
