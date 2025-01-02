import cv2
def get_available_cameras():
    """
    Detect and return a list of available camera devices.
    Returns a list of tuples containing (camera_index, camera_name)
    """
    available_cameras = []
    max_cameras_to_check = 10  # Adjust this number based on your needs
    
    for i in range(max_cameras_to_check):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            # Try to get the camera name/description
            # Note: This might not work on all systems/cameras
            ret, _ = cap.read()
            if ret:
                camera_name = f"Camera {i}"
                try:
                    # Attempt to get camera name (may not work on all systems)
                    backend = cv2.CAP_ANY
                    camera_name = cap.getBackendName()
                except:
                    pass
                available_cameras.append((i, camera_name))
            cap.release()
    
    return available_cameras

def initialize_camera(camera_index):
    """
    Initialize a camera with the given index and set properties for optimal performance.
    Returns the configured camera capture object.
    """
    cap = cv2.VideoCapture(camera_index)
    
    if cap.isOpened():
        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Adjust resolution as needed
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)  # Set desired frame rate
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)  # Enable autofocus if available
        
        # Additional properties can be set based on your needs
        # cap.set(cv2.CAP_PROP_BRIGHTNESS, 1.0)
        # cap.set(cv2.CAP_PROP_CONTRAST, 1.0)
        
    return cap

def capture_frame_from_camera(camera_index):
    """
    Capture a single frame from the specified camera.
    Returns the captured frame in RGB format.
    """
    cap = initialize_camera(camera_index)
    frame = None
    
    if cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    cap.release()
    return frame



get_available_cameras()