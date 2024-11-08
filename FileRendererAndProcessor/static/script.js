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
        const preprocessingOptions = [
            { value: 'resize', text: 'Resize' },
            { value: 'normalize', text: 'Normalize' },
            { value: 'grayscale', text: 'Grayscale' },
            { value: 'crop', text: 'Crop' }
        ];
        const augmentationOptions = [
            { value: 'rotate', text: 'Rotate' },
            { value: 'flip', text: 'Flip' },
            { value: 'blur', text: 'Blur' },
            { value: 'brightness', text: 'Brightness' }
        ];
        
        addOptionsToDropdown(preprocessingDropdown, preprocessingOptions);
        addOptionsToDropdown(augmentationDropdown, augmentationOptions);
    } 
    else if (fileType.startsWith('audio/')) {
        // Audio file options
        const preprocessingOptions = [
            { value: 'normalize_audio', text: 'Normalize Audio' },
            { value: 'trim_silence', text: 'Trim Silence' },
            { value: 'resample', text: 'Resample' },
            { value: 'noise_reduction', text: 'Noise Reduction' }
        ];
        const augmentationOptions = [
            { value: 'time_stretch', text: 'Time Stretch' },
            { value: 'pitch_shift', text: 'Pitch Shift' },
            { value: 'add_noise', text: 'Add Noise' },
            { value: 'reverb', text: 'Add Reverb' }
        ];
        
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
    const output = document.getElementById('output');
    
    if (fileInput.files.length > 0) {
        const file = fileInput.files[0];
        updateDropdownOptions(file.type);
        showFileName();
        enablePreprocessButton();
        
        try {
            if (file.type.startsWith('text/')) {
                // For text files
                const text = await file.text();
                output.innerHTML = `
                    <h3>Uploaded Content:</h3>
                    <div class="text-block">
                        <p><strong>Original Text:</strong></p>
                        <pre>${text}</pre>
                    </div>
                `;
            } else if (file.type.startsWith('image/')) {
                // For image files
                const imageUrl = URL.createObjectURL(file);
                output.innerHTML = `
                    <h3>Uploaded Content:</h3>
                    <div class="image-container">
                        <p><strong>Original Image:</strong></p>
                        <img src="${imageUrl}" alt="Uploaded image" style="max-width: 300px;">
                    </div>
                `;
            } else if (file.type.startsWith('audio/')) {
                // Audio file handling
                const audioUrl = URL.createObjectURL(file);
                output.innerHTML = `
                    <h3>Uploaded Content:</h3>
                    <div class="audio-container">
                        <p><strong>Original Audio:</strong></p>
                        <audio controls>
                            <source src="${audioUrl}" type="${file.type}">
                            Your browser does not support the audio element.
                        </audio>
                        <div class="audio-info">
                            <p><strong>File Name:</strong> ${file.name}</p>
                            <p><strong>File Size:</strong> ${(file.size / 1024 / 1024).toFixed(2)} MB</p>
                            <p><strong>File Type:</strong> ${file.type}</p>
                        </div>
                    </div>
                `;
            }
        } catch (error) {
            output.textContent = 'Error reading file: ' + error.message;
        }
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
    const output = document.getElementById('output');
    const fileInput = document.getElementById('fileInput');
    
    try {
        let result = null;
        const isImage = fileInput.files[0].type.startsWith('image/');
        const isAudio = fileInput.files[0].type.startsWith('audio/');
        
        // First do preprocessing if selected
        if (preprocessingType) {
            const formData = new FormData();
            formData.append('file', fileInput.files[0]);
            formData.append('preprocessing_type', preprocessingType.toLowerCase());
            
            const response = await fetch('/api/preprocess', {
                method: 'POST',
                body: formData
            });
            
            result = await response.json();
        }
        
        // Then do augmentation if selected
        if (augmentationType) {
            const formData = new FormData();
            
            if (result && result.processed_text) {
                // For text: use the preprocessed text
                const preprocessedBlob = new Blob([result.processed_text], { type: 'text/plain' });
                formData.append('file', preprocessedBlob, 'preprocessed.txt');
            } else if (result && result.processed_image_url) {
                // For images: fetch the processed image and use it
                const processedImage = await fetch(result.processed_image_url);
                const processedBlob = await processedImage.blob();
                formData.append('file', processedBlob, 'processed_image.jpg');
            } else {
                formData.append('file', fileInput.files[0]);
            }
            
            formData.append('augmentation_type', augmentationType.toLowerCase());
            
            const response = await fetch('/api/augment', {
                method: 'POST',
                body: formData
            });
            
            const augmentResult = await response.json();
            result = { ...result, ...augmentResult };
        }
        
        // Display the results
        if (result) {
            if (isImage) {
                output.innerHTML = `
                    <h3>Processing Results:</h3>
                    <div class="image-results">
                        <div class="image-container">
                            <p><strong>Original Image:</strong></p>
                            <img src="${URL.createObjectURL(fileInput.files[0])}" alt="Original">
                        </div>
                        ${result.processed_image_url ? `
                            <div class="image-container">
                                <p><strong>Preprocessed Image:</strong></p>
                                <img src="${result.processed_image_url}" alt="Preprocessed">
                                <p><strong>Operation:</strong> ${result.operation}</p>
                            </div>
                        ` : ''}
                        ${result.augmented_image_url ? `
                            <div class="image-container">
                                <p><strong>Augmented Image:</strong></p>
                                <img src="${result.augmented_image_url}" alt="Augmented">
                                <p><strong>Operation:</strong> ${result.operation}</p>
                            </div>
                        ` : ''}
                    </div>
                `;
            } 
            else if (isAudio) {
                output.innerHTML = `
                    <h3>Processing Results:</h3>
                    <div class="audio-results">
                        <div class="audio-container">
                            <p><strong>Original Audio:</strong></p>
                            <audio controls>
                                <source src="${URL.createObjectURL(fileInput.files[0])}" type="${fileInput.files[0].type}">
                            </audio>
                        </div>
                        
                        ${result.processed_audio_url ? `
                            <div class="audio-container">
                                <p><strong>Preprocessed Audio:</strong></p>
                                <audio controls>
                                    <source src="${result.processed_audio_url}" type="audio/wav">
                                </audio>
                                <p><strong>Operation:</strong> ${result.operation}</p>
                            </div>
                        ` : ''}
                        
                        ${result.augmented_audio_url ? `
                            <div class="audio-container">
                                <p><strong>Augmented Audio:</strong></p>
                                <audio controls>
                                    <source src="${result.augmented_audio_url}" type="audio/wav">
                                </audio>
                                <p><strong>Operation:</strong> ${result.operation}</p>
                            </div>
                        ` : ''}
                    </div>
                `;
            }
            else {
                output.innerHTML = `
                    <h3>Processing Results:</h3>
                    <div class="text-results">
                        <div class="text-block">
                            <p><strong>Original Text:</strong></p>
                            <pre>${result.original_text || ''}</pre>
                        </div>
                        
                        ${result.processed_text ? `
                            <div class="text-block">
                                <p><strong>Preprocessed Text:</strong></p>
                                <pre>${result.processed_text}</pre>
                                ${result.removed_words ? `<p><strong>Words Removed:</strong> ${result.removed_words}</p>` : ''}
                                ${result.operation ? `<p><strong>Operation:</strong> ${result.operation}</p>` : ''}
                            </div>
                        ` : ''}
                        
                        ${result.augmented_text ? `
                            <div class="text-block">
                                <p><strong>Augmented Text:</strong></p>
                                <pre>${result.augmented_text}</pre>
                                ${result.words_replaced ? `<p><strong>Words Replaced:</strong> ${result.words_replaced}</p>` : ''}
                                ${result.words_inserted ? `<p><strong>Words Inserted:</strong> ${result.words_inserted}</p>` : ''}
                            </div>
                        ` : ''}
                    </div>
                `;
            }
        }
        
    } catch (error) {
        output.textContent = 'Error processing file: ' + error.message;
    }
}

// Update handleDropdownChange to allow both preprocessing and augmentation
function handleDropdownChange(dropdownId) {
    // Remove the code that clears the other dropdown
    // This allows both preprocessing and augmentation to be selected
}