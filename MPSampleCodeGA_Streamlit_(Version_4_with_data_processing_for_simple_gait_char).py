import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from scipy.interpolate import UnivariateSpline
from scipy.fftpack import fft, ifft
import scipy.signal as sps
import tempfile # To handle temporary file storage

###### SECTION I: COMPUTER VISION -MEDIAPIPE POSE CODE BASE ######
# Function to detect and replace sudden jumps in the signal data
def replace_jump(signal, threshold):
    for n in range(1, len(signal)):
        if n >= 0 and n < len(signal) - 1:
            # If a jump is detected based on the threshold, interpolate the value
            if np.abs(signal[n] - signal[n - 1]) > threshold * np.abs(signal[n + 1] - signal[n - 1]):
                previous_value = signal[n - 1]
                next_value = signal[n + 1]
                interpolated_value = (previous_value + next_value) / 2
                signal[n] = interpolated_value  # Replace the jump value with the interpolated value
    return signal

# Function to calculate the angle between three points (a, b, c)
def calculate_angle(a, b, c):
    a = np.array(a)  # First point
    b = np.array(b)  # Middle point
    c = np.array(c)  # End point

    # Calculate the angle in degrees
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = 180 - np.abs(radians * 180.0 / np.pi)
    
    return angle

# Function to smooth a sequence of angles using spline interpolation
def smooth_angles(angles, smoothing_factor=5):
    frame_numbers = np.array(range(len(angles)))
    spline = UnivariateSpline(frame_numbers, angles, k=5, s=smoothing_factor)
    return spline(frame_numbers)

# Main function to process the video and analyze gait
def process_video(input_path, output_path, output_path2):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(smooth_landmarks=True)

    # Open the input video file
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    # Create VideoWriter objects for output videos
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    out2 = cv2.VideoWriter(output_path2, fourcc, fps, (frame_width, frame_height))

    # Lists to store raw and processed data
    angles_data = []
    RSHDx, RSHDy = [], []
    RHIPx, RHIPy = [], []
    RKNEx, RKNEy = [], []
    RANKx, RANKy = [], []
    RHEEx, RHEEy = [], []
    RTOEx, RTOEy = [], []

    frames_with_landmarks = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the BGR image to RGB for processing with MediaPipe
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_image)  # Process the RGB image
        bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)  # Convert back to BGR for display

        if results.pose_landmarks:
            frames_with_landmarks.append(frame)
            # (Existing code to process landmarks and draw them on the frame)
            landmarks = results.pose_landmarks.landmark
            mp.solutions.drawing_utils.draw_landmarks(
                bgr_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate joint angles for the right hip, knee, and ankle
            hip = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]
            )
            
            knee = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y]
            )

            ankle = calculate_angle(
                [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y],
                [landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y]
            )

            # Store raw landmark data
            RSHDx.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x)
            RSHDy.append(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y)
            RHIPx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x)
            RHIPy.append(landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y)
            RKNEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x)
            RKNEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y)
            RANKx.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x)
            RANKy.append(landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y)
            RHEEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].x)
            RHEEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_HEEL.value].y)
            RTOEx.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].x)
            RTOEy.append(landmarks[mp_pose.PoseLandmark.RIGHT_FOOT_INDEX.value].y)

            # Append calculated angles to the list
            angles_data.append({
                'Time (ms)': (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps,
                'Hip Flexion-Extension (deg)': hip,
                'Knee Flexion-Extension (deg)': knee,
                'Ankle Dorsiflexion-Plantarflexion (deg)': ankle
            })

        # Write the processed frame with landmarks to the output video
        out.write(bgr_image)

    # Apply the jump replacement filter to smooth out the data
    fRSHDx = replace_jump(RSHDx, 1.5)
    fRSHDy = replace_jump(RSHDy, 1.5)
    fRHIPx = replace_jump(RHIPx, 1.5)
    fRHIPy = replace_jump(RHIPy, 1.5)
    fRKNEx = replace_jump(RKNEx, 1.5)
    fRKNEy = replace_jump(RKNEy, 1.5)
    fRANKx = replace_jump(RANKx, 1.5)
    fRANKy = replace_jump(RANKy, 1.5)
    fRHEEx = replace_jump(RHEEx, 1.5)
    fRHEEy = replace_jump(RHEEy, 1.5)
    fRTOEx = replace_jump(RTOEx, 1.5)
    fRTOEy = replace_jump(RTOEy, 1.5)
    
    # To create filtered output video
    i = 0  # Frame counter
    for i, frame2 in enumerate(frames_with_landmarks):

        # Define circle radius
        rad = 5

        # Define circle parameters for multiple landmarks
        circles = [
            {'radius': rad, 'center_x': fRSHDx[i] * frame_width, 'center_y': fRSHDy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRHIPx[i] * frame_width, 'center_y': fRHIPy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRKNEx[i] * frame_width, 'center_y': fRKNEy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRANKx[i] * frame_width, 'center_y': fRANKy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRHEEx[i] * frame_width, 'center_y': fRHEEy[i] * frame_height, 'color': (0, 255, 0)},
            {'radius': rad, 'center_x': fRTOEx[i] * frame_width, 'center_y': fRTOEy[i] * frame_height, 'color': (0, 255, 0)},
        ]

        # Draw circles on the image
        for circle in circles:
            markedframe = cv2.circle(frame2, (int(circle['center_x']), int(circle['center_y'])), circle['radius'], circle['color'], -1)
        
        # Draw lines connecting the circles
        for j in range(0, len(circles) - 1):
            start_point = (int(circles[j]['center_x']), int(circles[j]['center_y']))
            end_point = (int(circles[j + 1]['center_x']), int(circles[j + 1]['center_y']))
            markedframe = cv2.line(frame2, start_point, end_point, (0, 255, 0), 1)

        # Write the processed frame with filtered landmarks to the output video
        out2.write(markedframe)
        i += 1  # Increment the frame counter

    # Release resources
    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()
    return RTOEy, fRTOEy, fps

###### SECTION II: DATA WRANGLING ######

def fourier_filter(data, sampling_freq, critical_freq):
    # Fourier Transform
    N = len(data)
    freqs = np.fft.fftfreq(N, d=1/sampling_freq)
    fft_values = fft(data)
    
    # Apply Low-Pass Filter
    fft_values[np.abs(freqs) > critical_freq] = 0  # Zero out high-frequency components
    
    # Inverse Fourier Transform
    filtered_data = np.real(ifft(fft_values))
    return filtered_data

# Function to extract peaks
def extract_peaks(data):
    # Find the indices of peaks
    peak_indices, _ = sps.find_peaks(data)
    # Collect the (x, y) tuples for peaks
    peaks = [(i, data[i]) for i in peak_indices]
    return peaks

###### SECTION III: STREAMLIT UI AND DATA PROCESSING ######

st.title("Gait Analysis Using Computer Vision")
st.write("This tool is powered by the ML solution MediaPipe Pose by Google. \n Try to keep the video within 20 seconds to prevent the application from loading forever.")

# File uploader
uploaded_file = st.file_uploader("Upload an MP4 File Containing a Person Walking. [Click to View an Example from YouTube](https://www.youtube.com/watch?v=Mol0lrRBy3g)", type=["mp4"])

if uploaded_file is not None:
    # Create temporary file paths
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4v") as input_tmp:
        input_tmp.write(uploaded_file.read())
        input_video_path = input_tmp.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4v") as output_tmp1:
        output_video_path = output_tmp1.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4v") as output_tmp2:
        output_video_path2 = output_tmp2.name

    # Process the video
    RTOEy, fRTOEy, fps = process_video(input_video_path, output_video_path, output_video_path2)

    # Display the processed video
    st.subheader("Processed Video with all Pose Landmarks")
    if output_video_path is not None:
        st.video(output_video_path)
    else:
        st.error

    st.subheader("Processed Video with Noise Corrected")
    st.write('Frames where landmarks are not detected are removed from the video.')
    if output_video_path is not None:
        st.video(output_video_path2)
    else:
        st.error

    st.subheader("Right Toe Y-Coordinate Analysis")
    st.write('Further analysis by tracking movement of the right toe.')
    # Plot of y-coordinates for the right toe.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fRTOEy, label="Filtered Data", color="red", marker='x')
    ax.set_title("Right Toe Y-Coordinate (RAW DATA)")
    ax.set_xlabel("Frame Index (Progression of Gait)")
    ax.set_ylabel("Y-Coordinate")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Apply Low Pass Filter through Fourier Transform to smoothen signal; max_value controlled by Nyquist Sampling Theorem
    st.write('A normal walking frequency is typically below 2.0 Hz')
    critical_freq = st.slider("Adjust Critical Frequency such that distinct peaks can be observed, use the new graph below to gauge.", min_value=0.1, max_value=fps / 2, value=1.5, step=0.1)
    fRTOEy = fourier_filter(fRTOEy, fps, critical_freq) # Update fRTOEy data

    # Plot of filtered y-coordinates for the right toe.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(fRTOEy, label="Filtered Data", color="red", marker='x')
    ax.set_title("Right Toe Y-Coordinate (FILTERED DATA)")
    ax.set_xlabel("Frame Index (Progression of Gait)")
    ax.set_ylabel("Y-Coordinate")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Extract peaks
    peaks = extract_peaks(fRTOEy)
    peak_df = pd.DataFrame(peaks, columns=["Frame Index", "Y-Coordinate"])

    # Streamlit UI for frame range selection
    st.subheader("Peak Detection and Gait Analysis")
    frame_min = st.number_input(f"Minimum frame index (0 ≤ Integer ≤ {len(fRTOEy)}):", min_value=0, max_value=len(fRTOEy) - 1, value=0)
    frame_max = st.number_input(f"Maximum frame index ({frame_min} ≤ Integer ≤ {len(fRTOEy)}):", min_value=0, max_value=len(fRTOEy) - 1, value=len(fRTOEy) - 1)

    # Filter peaks based on selected frame range
    filtered_peaks = peak_df[(peak_df["Frame Index"] >= frame_min) & (peak_df["Frame Index"] <= frame_max)]

    # Display table of filtered peaks
    if st.button('Click to display table of peaks'):
        st.write("Peaks within the selected range:")
        st.table(filtered_peaks)

    # Calculate gait characteristics if there are enough peaks
    if len(filtered_peaks) >= 2:
        # Calculate step length (difference between consecutive peaks)
        filtered_peaks['Step Length'] = filtered_peaks["Frame Index"].diff().fillna(0)

        # Calculate stride duration (difference between every second peak, assuming stride involves 2 steps)
        stride_durations = [
            filtered_peaks["Frame Index"].iloc[i + 2] - filtered_peaks["Frame Index"].iloc[i]
            for i in range(len(filtered_peaks) - 2)
        ]

        # Display calculated characteristics
        st.subheader("Gait Characteristics")
        st.write("Here are some Gait Characteristics calculated using data within the time frame you have selected.")
        step_duration = round(filtered_peaks['Step Length'].mean(), 2)
        st.write(f"**Average Step Duration**: {step_duration} frames ({round(step_duration/fps,3)} seconds)")
        cycle_time = round(np.mean(stride_durations), 2)
        st.write(f"**Average Cycle Time**: {cycle_time} frames ({round(cycle_time/fps,3)} seconds)")
        cadence = round(60 / (step_duration/fps),1) # Number of steps per minute
        st.write(f"**Cadence**: {cadence} steps / minute")
    else:
        st.error("Not enough peaks in the selected range to calculate gait characteristics.")

# Instructions to Refresh the page
st.info('Refresh the Page or click on \'Browse Files\' again to process the next video.')
st.warning('Note: Download your processed videos before refreshing if you want to keep them.')
