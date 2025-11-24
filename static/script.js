/* static/script.js */

// Jab page load ho jaye
document.addEventListener('DOMContentLoaded', function() {
    console.log("SafeMask AI System Loaded Successfully.");
});

// Full Screen Mode function
function toggleFullScreen() {
    if (!document.fullscreenElement) {
        document.documentElement.requestFullscreen();
    } else {
        if (document.exitFullscreen) {
            document.exitFullscreen();
        }
    }
}

// Stop Button Action
function stopSystem() {
    let confirmAction = confirm("Are you sure you want to stop the monitoring system?");
    if (confirmAction) {
        alert("System Stopped. Logs saved locally.");
        // Yahan tum chaho to page reload kar sakte ho ya video hide kar sakte ho
        // location.reload();
    }
}

// View Logs Action
function viewLogs() {
    alert("Opening System Logs... (This feature can be linked to a database)");
}