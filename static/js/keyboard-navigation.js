document.addEventListener('DOMContentLoaded', function() {
    // Track the currently focused video container
    let currentFocusedVideo = null;
    
    // Map classification types to keyboard shortcuts
    const shortcutMap = {
        'smoke': 'm',
        'steam': 't',
        'shadow': 'h',
        'other_motion': 'o',
        'none': 'n'
    };
    
    // Add keyboard shortcut indicators to labels
    function addShortcutIndicators() {
        const labels = document.querySelectorAll('.checkbox-item label');
        labels.forEach(label => {
            const text = label.textContent.trim();
            const inputEl = document.getElementById(label.getAttribute('for'));
            if (!inputEl) return;
            
            const classType = inputEl.value;
            const shortcutKey = shortcutMap[classType];
            
            if (shortcutKey) {
                // Find the position of the shortcut letter in the text
                const lowerText = text.toLowerCase();
                const position = lowerText.indexOf(shortcutKey);
                if (position >= 0) {
                    // Split and reassemble the text with the shortcut span
                    const before = text.substring(0, position);
                    const letter = text.substring(position, position + 1);
                    const after = text.substring(position + 1);
                    
                    label.innerHTML = before + '<span class="key-shortcut">' + letter + '</span>' + after;
                }
            }
        });
    }
    
    // Set focus to a specific video container
    function setFocus(container) {
        if (!container) return;
        
        // Remove focus from current container
        if (currentFocusedVideo) {
            currentFocusedVideo.classList.remove('focused');
        }
        
        // Set focus to new container
        currentFocusedVideo = container;
        container.classList.add('focused');
        
        // Scroll into view if needed
        container.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
    
    // Find the next video with no checkboxes set
    function findNextUnclassifiedVideo(forward = true) {
        const containers = Array.from(document.querySelectorAll('.video-container'));
        if (!containers.length) return null;
        
        let currentIndex = -1;
        if (currentFocusedVideo) {
            currentIndex = containers.indexOf(currentFocusedVideo);
        }
        
        let step = forward ? 1 : -1;
        let index = currentIndex;
        
        for (let i = 0; i < containers.length; i++) {
            index = (index + step + containers.length) % containers.length;
            
            const videoContainer = containers[index];
            const checkboxes = videoContainer.querySelectorAll('input[type="checkbox"]:checked');
            
            if (checkboxes.length === 0) {
                return videoContainer;
            }
        }
        
        // If no unclassified videos found, return the next video in sequence
        return containers[(currentIndex + step + containers.length) % containers.length];
    }
    
    // Navigate to the next or previous video
    function navigateVideo(forward = true) {
        const containers = Array.from(document.querySelectorAll('.video-container'));
        if (!containers.length) return;
        
        let currentIndex = 0;
        if (currentFocusedVideo) {
            currentIndex = containers.indexOf(currentFocusedVideo);
            currentIndex = (currentIndex + (forward ? 1 : -1) + containers.length) % containers.length;
        }
        
        setFocus(containers[currentIndex]);
    }
    
    // Toggle a checkbox in the focused container
    function toggleClassificationByKey(classType) {
        if (!currentFocusedVideo) return;
        
        const checkbox = currentFocusedVideo.querySelector(`input[value="${classType}"]`);
        if (checkbox) {
            checkbox.checked = !checkbox.checked;
            
            // Trigger change event to ensure the existing code handles the update
            const event = new Event('change', { bubbles: true });
            checkbox.dispatchEvent(event);
        }
    }
    
    // Initialize focus handlers for videos and checkboxes
    function initializeFocusHandlers() {
        const videoContainers = document.querySelectorAll('.video-container');
        
        videoContainers.forEach(container => {
            // Focus on click (videos and container)
            container.addEventListener('click', function() {
                setFocus(container);
            });
            
            // Focus on checkbox click too
            const checkboxes = container.querySelectorAll('input[type="checkbox"]');
            checkboxes.forEach(checkbox => {
                checkbox.addEventListener('click', function(e) {
                    e.stopPropagation();
                    setFocus(container);
                });
            });
        });
    }
    
    // Initialize keyboard event handling
    function initializeKeyboardEvents() {
        document.addEventListener('keydown', function(e) {
            // Skip if in text input
            if (e.target.tagName === 'INPUT' && e.target.type === 'text' || 
                e.target.tagName === 'TEXTAREA') {
                return;
            }
            
            switch(e.key) {
                case 'Tab':
                    // Tab to next unclassified video
                    e.preventDefault();
                    const videoContainer = findNextUnclassifiedVideo(!e.shiftKey);
                    setFocus(videoContainer);
                    break;
                    
                case 'ArrowRight':
                    // Navigate to next video
                    e.preventDefault();
                    navigateVideo(true);
                    break;
                    
                case 'ArrowLeft':
                    // Navigate to previous video
                    e.preventDefault();
                    navigateVideo(false);
                    break;
                    
                default:
                    // Check for classification shortcuts
                    for (const classType in shortcutMap) {
                        if (e.key.toLowerCase() === shortcutMap[classType]) {
                            toggleClassificationByKey(classType);
                            break;
                        }
                    }
                    break;
            }
        });
    }
    
    // Init
    addShortcutIndicators();
    initializeFocusHandlers();
    initializeKeyboardEvents();
    
    // Set initial focus to the first video - ensure this runs AFTER all other scripts
    // The jQuery init function might run after this DOMContentLoaded handler
    setTimeout(() => {
        const firstVideo = document.querySelector('.video-container');
        if (firstVideo && !currentFocusedVideo) {
            setFocus(firstVideo);
        }
    }, 500);
});

// Also add an event listener for window load to ensure initial focus works
window.addEventListener('load', function() {
    // Check if any video is already focused
    const focusedVideo = document.querySelector('.video-container.focused');
    if (!focusedVideo) {
        // If no video is focused yet, focus the first one
        const firstVideo = document.querySelector('.video-container');
        if (firstVideo) {
            // Get the setFocus function from the DOMContentLoaded scope
            const focusEvent = new CustomEvent('set-initial-focus', {
                detail: { container: firstVideo }
            });
            document.dispatchEvent(focusEvent);
        }
    }
});

// Listen for the custom event
document.addEventListener('set-initial-focus', function(e) {
    const container = e.detail.container;
    if (container) {
        // Remove focus from any currently focused container
        const currentFocused = document.querySelector('.video-container.focused');
        if (currentFocused) {
            currentFocused.classList.remove('focused');
        }
        
        // Set focus to the container
        container.classList.add('focused');
        
        // Scroll into view if needed
        container.scrollIntoView({
            behavior: 'smooth',
            block: 'center'
        });
    }
});
