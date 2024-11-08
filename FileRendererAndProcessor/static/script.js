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

function updateDropdownOptions(fileType) {
    const preprocessingDropdown = document.getElementById('preprocessing');
    const augmentationDropdown = document.getElementById('augmentation');
    
    // Clear existing options
    preprocessingDropdown.innerHTML = '<option value="">Select Preprocessing</option>';
    augmentationDropdown.innerHTML = '<option value="">Select Augmentation</option>';
    
    if (fileType.startsWith('text/')) {
        // Text file options
        const preprocessingOptions = [
            { value: 'remove_stopwords', text: 'Remove Stopwords' },
            { value: 'lowercase', text: 'Lowercase' },
            { value: 'tokenize', text: 'Tokenize' },
            { value: 'remove_punctuation', text: 'Remove Punctuation' }
        ];
        const augmentationOptions = [
            { value: 'synonym_replace', text: 'Synonym Replace' },
            { value: 'back_translation', text: 'Back Translation' },
            { value: 'random_insert', text: 'Random Insert' }
        ];
        
        addOptionsToDropdown(preprocessingDropdown, preprocessingOptions);
        addOptionsToDropdown(augmentationDropdown, augmentationOptions);
    } 
    else if (fileType.startsWith('image/')) {
        // Image file options
        const preprocessingOptions = ['Resize', 'Normalize', 'Grayscale', 'Crop'];
        const augmentationOptions = ['Rotate', 'Flip', 'Blur', 'Brightness'];
        
        addOptionsToDropdown(preprocessingDropdown, preprocessingOptions);
        addOptionsToDropdown(augmentationDropdown, augmentationOptions);
    } 
    else if (fileType.startsWith('audio/')) {
        // Audio file options
        const preprocessingOptions = ['Normalize Audio', 'Resample', 'Trim Silence'];
        const augmentationOptions = ['Time Stretch', 'Pitch Shift', 'Add Noise'];
        
        addOptionsToDropdown(preprocessingDropdown, preprocessingOptions);
        addOptionsToDropdown(augmentationDropdown, augmentationOptions);
    }
}

function addOptionsToDropdown(dropdown, options) {
    options.forEach(option => {
        const optionElement = document.createElement('option');
        if (typeof option === 'object') {
            optionElement.value = option.value;
            optionElement.textContent = option.text;
        } else {
            optionElement.value = option.toLowerCase().replace(' ', '_');
            optionElement.textContent = option;
        }
        dropdown.appendChild(optionElement);
    });
}

async function handleFileSelect() {
    const fileInput = document.getElementById('fileInput');
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        updateDropdownOptions(file.type);  // Update dropdowns based on file type
        showFileName();
        await uploadFile();
        enablePreprocessButton();
    }
}

function enablePreprocessButton() {
    const preprocessBtn = document.getElementById('preprocessBtn');
    preprocessBtn.disabled = false;
    preprocessBtn.classList.remove('disabled');
}

async function preprocess() {
    const fileInput = document.getElementById('fileInput');
    const preprocessingType = document.getElementById('preprocessing').value;
    const output = document.getElementById('output');
    
    if (!fileInput.files.length) {
        output.textContent = 'Please upload a file first';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('preprocessing_type', preprocessingType);

    try {
        const response = await fetch('/api/preprocess', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        // Display results based on preprocessing type
        if (result.tokens) {
            output.innerHTML = `
                <h3>Tokenization Results:</h3>
                <p><strong>Original Text:</strong></p>
                <pre>${result.original_text}</pre>
                <p><strong>Tokens:</strong></p>
                <pre>${JSON.stringify(result.tokens, null, 2)}</pre>
                <p><strong>Token Count:</strong> ${result.token_count}</p>
            `;
        } else {
            output.innerHTML = `
                <h3>Preprocessing Results:</h3>
                <p><strong>Original Text:</strong></p>
                <pre>${result.original_text}</pre>
                <p><strong>Processed Text:</strong></p>
                <pre>${result.processed_text}</pre>
                ${result.removed_words ? `<p><strong>Stopwords Removed:</strong> ${result.removed_words}</p>` : ''}
                ${result.operation ? `<p><strong>Operation:</strong> ${result.operation}</p>` : ''}
            `;
        }
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

async function augment() {
    const fileInput = document.getElementById('fileInput');
    const augmentationType = document.getElementById('augmentation').value;
    const output = document.getElementById('output');
    
    if (!fileInput.files.length) {
        output.textContent = 'Please upload a file first';
        return;
    }

    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    formData.append('augmentation_type', augmentationType);

    try {
        const response = await fetch('/api/augment', {
            method: 'POST',
            body: formData
        });

        const result = await response.json();
        
        output.innerHTML = `
            <h3>Augmentation Results:</h3>
            <p><strong>Original Text:</strong></p>
            <pre>${result.original_text}</pre>
            <p><strong>Augmented Text:</strong></p>
            <pre>${result.augmented_text}</pre>
            <p><strong>Words Replaced:</strong> ${result.words_replaced}</p>
        `;
    } catch (error) {
        output.textContent = 'Error augmenting file: ' + error.message;
    }
}

async function handleAction() {
    const preprocessingType = document.getElementById('preprocessing').value;
    const augmentationType = document.getElementById('augmentation').value;
    
    if (preprocessingType && !augmentationType) {
        // Only preprocessing is selected
        await preprocess();
    } else if (augmentationType && !preprocessingType) {
        // Only augmentation is selected
        await augment();
    } else if (preprocessingType && augmentationType) {
        // Both are selected - you might want to show an error or handle this case
        const output = document.getElementById('output');
        output.textContent = 'Please select either preprocessing OR augmentation, not both';
    } else {
        // Neither is selected
        const output = document.getElementById('output');
        output.textContent = 'Please select a preprocessing or augmentation option';
    }
}

// Add this new function
function handleDropdownChange(dropdownId) {
    const preprocessing = document.getElementById('preprocessing');
    const augmentation = document.getElementById('augmentation');
    
    if (dropdownId === 'preprocessing' && preprocessing.value) {
        augmentation.value = '';  // Clear augmentation if preprocessing is selected
    } else if (dropdownId === 'augmentation' && augmentation.value) {
        preprocessing.value = '';  // Clear preprocessing if augmentation is selected
    }
}