#!/usr/bin/env node

console.log("Sinhala Fake News CLI Checker");
console.log("Usage: node check_cli.js --text '...'");

const args = process.argv.slice(2);
if (args.length === 0) {
    console.log("No text provided.");
    process.exit(1);
}

// Placeholder for API call
console.log(`Checking text: ${args[0]}`);
console.log("Verdict: NEEDS MORE EVIDENCE (Mock)");
