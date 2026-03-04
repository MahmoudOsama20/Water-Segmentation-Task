const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const spinner = document.getElementById("spinner");
const originalImage = document.getElementById("original-image");
const resultImage = document.getElementById("result-image");

dropZone.addEventListener("click", () => fileInput.click());

dropZone.addEventListener("dragover", e => {
    e.preventDefault();
    dropZone.style.background = "rgba(255,255,255,0.2)";
});

dropZone.addEventListener("dragleave", () => {
    dropZone.style.background = "transparent";
});

dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.style.background = "transparent";
    handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener("change", () => {
    handleFile(fileInput.files[0]);
});

function handleFile(file) {

    if (!file) return;

    const formData = new FormData();
    formData.append("file", file);

    spinner.style.display = "block";

    fetch("/predict", {
        method: "POST",
        body: formData
    })
    .then(response => response.json())
    .then(data => {

        spinner.style.display = "none";

        if (data.error) {
            alert(data.error);
            return;
        }

        originalImage.src = "data:image/png;base64," + data.original;
        resultImage.src = "data:image/png;base64," + data.mask;
    })
    .catch(error => {
        spinner.style.display = "none";
        alert("Prediction failed.");
    });
}