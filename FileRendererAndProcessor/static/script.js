function showFileName() {
    const input = document.getElementById('fileInput');
    const text = document.getElementById('uploadText');
    
    if (input.files.length > 0) {
        text.textContent = input.files[0].name;
        text.classList.add('file-selected');
    } else {
        text.textContent = 'Click or drag files here to upload';
        text.classList.remove('file-selected');
    }
}

async function handleFileSelect() {
    showFileName();
    await uploadFile();
    enablePreprocessButton();
}

function enablePreprocessButton() {
    const preprocessBtn = document.getElementById('preprocessBtn');
    preprocessBtn.disabled = false;
    preprocessBtn.classList.remove('disabled');
}

async function preprocess() {
    const output = document.getElementById('output');
    try {
        const response = await fetch('/api/preprocess', {
            method: 'POST'
        });
        const result = await response.json();
        output.textContent = JSON.stringify(result, null, 2);
    } catch (error) {
        output.textContent = 'Error preprocessing file: ' + error.message;
    }
}

async function uploadFile() {
    const fileInput = document.getElementById('fileInput');
    const output = document.getElementById('output');
    
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const result = await response.json();

            if (result.content) {
                output.textContent = result.content;
            } else if (result.audio_url) {
                output.innerHTML = `<audio controls src="${result.audio_url}"></audio>`;
            } else if (response.headers.get('content-type').startsWith('image/')) {
                const imageUrl = URL.createObjectURL(await response.blob());
                output.innerHTML = `<img src="${imageUrl}" style="max-width: 100%;">`;
            } else {
                output.textContent = JSON.stringify(result, null, 2);
            }
        } catch (error) {
            output.textContent = 'Error uploading file: ' + error.message;
        }
    }
}