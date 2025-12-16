const imageInput = document.getElementById('imageInput');
const imagePreview = document.getElementById('imagePreview');
const resultModal = document.getElementById('resultModal');
const resultTitle = document.getElementById('resultTitle');
const resultText = document.getElementById('resultText');
const confidenceBar = document.getElementById('confidenceBar');

let selectedFile = null;

// Handle File Selection
imageInput.addEventListener('change', function (event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        const reader = new FileReader();
        reader.onload = function (e) {
            imagePreview.innerHTML = `<img src="${e.target.result}" class="preview" alt="Preview">`;
        }
        reader.readAsDataURL(file);
    }
});

// Handle Verification
async function verifyImage() {
    if (!selectedFile) {
        alert("Please upload an image first!");
        return;
    }

    // Show loading state (optional)
    const verifyBtn = document.querySelector('.btn-verify');
    const originalText = verifyBtn.innerHTML;
    verifyBtn.innerHTML = "Checking...";
    verifyBtn.disabled = true;

    const formData = new FormData();
    formData.append('file', selectedFile);

    try {
        const response = await fetch('http://127.0.0.1:8000/predict', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            throw new Error(`Error: ${response.statusText}`);
        }

        const data = await response.json();
        showResult(data);

    } catch (error) {
        console.error('Error:', error);
        alert("Failed to verify image. Make sure the API is running.");
    } finally {
        verifyBtn.innerHTML = originalText;
        verifyBtn.disabled = false;
    }
}

function showResult(data) {
    resultModal.classList.remove('hidden');

    // Using data.label as per the schema fix
    const isAlpaca = data.label.toLowerCase() === 'alpaca';
    const confidencePercent = Math.round(data.confidence * 100);

    if (isAlpaca) {
        resultTitle.innerText = "IT'S AN ALPACA! 🦙";
        resultTitle.style.color = "#7CFC00";
    } else {
        resultTitle.innerText = "NOT AN ALPACA ❌";
        resultTitle.style.color = "#FF4500";
    }

    resultText.innerText = `Confidence: ${confidencePercent}%`;
    confidenceBar.style.width = `${confidencePercent}%`;

    // Change bar color based on confidence
    if (confidencePercent > 80) {
        confidenceBar.style.backgroundColor = "#7CFC00";
    } else if (confidencePercent > 50) {
        confidenceBar.style.backgroundColor = "#FFD700";
    } else {
        confidenceBar.style.backgroundColor = "#FF4500";
    }
}

function closeModal() {
    resultModal.classList.add('hidden');
}

// Close modal when clicking outside
window.onclick = function (event) {
    if (event.target == resultModal) {
        closeModal();
    }
}