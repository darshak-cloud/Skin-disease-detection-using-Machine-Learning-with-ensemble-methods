let video = document.getElementById('webcam');
let canvas = document.getElementById('canvas');
let context = canvas.getContext('2d');

// Start the webcam
function startWebcam() {
    navigator.mediaDevices.getUserMedia({ video: true })
    .then(function(stream) {
        video.srcObject = stream;
        video.play();
    })
    .catch(function(err) {
        console.error("Error accessing webcam: " + err);
        alert("Unable to access webcam. Please ensure it is connected and try again.");
    });
}

// Capture an image from the webcam
function captureImage() {
    if (video.srcObject) {
        // Set canvas size to match video dimensions
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        // Draw the video frame to the canvas
        context.drawImage(video, 0, 0, canvas.width, canvas.height);

        // Convert canvas to base64 data URL (JPEG format)
        let dataURL = canvas.toDataURL('image/jpeg');

        // Set the hidden input field with the image data
        document.getElementById('image_data').value = dataURL;

        // Submit the form
        document.getElementById('webcam-form').submit();
    } else {
        alert('No webcam feed available.');
    }
}

// Convert data URL to Blob (if needed for other purposes)
function dataURLtoBlob(dataURL) {
    let byteString = atob(dataURL.split(',')[1]);
    let mimeString = dataURL.split(',')[0].split(':')[1].split(';')[0];
    let ab = new ArrayBuffer(byteString.length);
    let ia = new Uint8Array(ab);

    for (let i = 0; i < byteString.length; i++) {
        ia[i] = byteString.charCodeAt(i);
    }

    return new Blob([ab], { type: mimeString });
}
