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
    LTOEx, LTOEy = [], []

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

            # Overlay angles as text on the frame
            text_hip = f"Right Hip: {hip:.1f} degrees"
            text_knee = f"Right Knee: {knee:.1f} degrees"
            text_ankle = f"Right Ankle: {ankle:.1f} degrees"

            rect_x, rect_y = 30, 10  # Top-left corner of the rectangle
            rect_width, rect_height = 310, 80  # Rectangle width and height
            cv2.rectangle(bgr_image, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1) # Background Rectangle
            cv2.putText(bgr_image, text_hip, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr_image, text_knee, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(bgr_image, text_ankle, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

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
            LTOEy.append(landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX.value].y)

            # Append calculated angles to the list
            angles_data.append({
                'Time (s)': (cap.get(cv2.CAP_PROP_POS_FRAMES) - 1) / fps,
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
    fLTOEy = replace_jump(LTOEy, 1.5)
    
    # To create filtered output video
    i = 0  # Frame counter
    for i, frame2 in enumerate(frames_with_landmarks):

        # Calculate joint angles for the right hip, knee, and ankle
        hip = calculate_angle(
            [fRSHDx[i], fRSHDy[i]],  # Right shoulder
            [fRHIPx[i], fRHIPy[i]],  # Right hip
            [fRKNEx[i], fRKNEy[i]]   # Right knee
        )

        knee = calculate_angle(
            [fRHIPx[i], fRHIPy[i]],  # Right hip
            [fRKNEx[i], fRKNEy[i]],  # Right knee
            [fRANKx[i], fRANKy[i]]   # Right ankle
        )

        ankle = calculate_angle(
            [fRKNEx[i], fRKNEy[i]],  # Right knee
            [fRANKx[i], fRANKy[i]],  # Right ankle
            [fRHEEx[i], fRHEEy[i]]   # Right heel
        )

        # Overlay angles as text on the filtered frame
        text_hip = f"Right Hip: {hip:.1f} degrees"
        text_knee = f"Right Knee: {knee:.1f} degrees"
        text_ankle = f"Right Ankle: {ankle:.1f} degrees"

        rect_x, rect_y = 30, 10  # Top-left corner of the rectangle
        rect_width, rect_height = 310, 80  # Rectangle width and height
        cv2.rectangle(frame2, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)  # Background Rectangle
        cv2.putText(frame2, text_hip, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, text_knee, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame2, text_ankle, (30, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Define circle radius
        rad = 5

        # Define circle and text overlay parameters for each joint
        joints = [
            {
                "name": f"{hip:.1f}",
                "x": fRHIPx[i] * frame_width,
                "y": fRHIPy[i] * frame_height,
            },
            {
                "name": f"{knee:.1f}",
                "x": fRKNEx[i] * frame_width,
                "y": fRKNEy[i] * frame_height,
            },
            {
                "name": f"{ankle:.1f}",
                "x": fRANKx[i] * frame_width,
                "y": fRANKy[i] * frame_height,
            },
        ]

        # Draw circles and text overlays for each joint
        for joint in joints:
            # Draw the circle
            frame2 = cv2.circle(
                frame2, (int(joint["x"]), int(joint["y"])), rad, (0, 255, 0), -1
            )

            # Draw the text slightly above and to the right of the circle
            text_offset_x = 10
            text_offset_y = -10
            # Background Rectangle
            rect_x, rect_y = int(joint["x"] + text_offset_x), int(joint["y"] + text_offset_y - 15)  # Top-left corner of the rectangle
            rect_width, rect_height = 50, 20  # Rectangle width and height
            cv2.rectangle(frame2, (rect_x, rect_y), (rect_x + rect_width, rect_y + rect_height), (0, 0, 0), -1)
            # Text
            frame2 = cv2.putText(
                frame2,
                joint["name"],
                (int(joint["x"] + text_offset_x), int(joint["y"] + text_offset_y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

        # Draw lines connecting the circles (same as before)
        # Define the joint connections
        connections = [
            (joints[0], joints[1]),  # Hip to Knee
            (joints[1], joints[2]),  # Knee to Ankle
        ]

        for start, end in connections:
            start_point = (int(start["x"]), int(start["y"]))
            end_point = (int(end["x"]), int(end["y"]))
            frame2 = cv2.line(frame2, start_point, end_point, (0, 255, 0), 2)

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

    # Convert data to a DataFrame for easier handling
    angles_data = pd.DataFrame(angles_data)

    # Release resources
    cap.release()
    out.release()
    out2.release()
    cv2.destroyAllWindows()
    return RTOEy, LTOEy, fRTOEy, fLTOEy, fps, angles_data

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
    # Collect the (index, value) tuples for peaks
    peaks = [(i, data[i]) for i in peak_indices]
    return peaks

# Function to extract troughs
def extract_troughs(data):
    # Find the indices of troughs by inverting the data
    trough_indices, _ = sps.find_peaks(-data)
    # Collect the (index, value) tuples for troughs
    troughs = [(i, data[i]) for i in trough_indices]
    return troughs

###### SECTION III: STREAMLIT UI AND DATA PROCESSING ######

st.title("Gait Analysis Using Computer Vision")
st.write("This tool is powered by the ML solution MediaPipe Pose by Google.")
st.write("Try to keep the video within 15 seconds to prevent the application from loading forever.")
st.write("The most accurate results shall be obtained from an ideal video containing a person **visible from head to toe** walking on a **treadmill**/with the **camera following the person** while walking [Click to View an Example from YouTube](https://www.youtube.com/watch?v=Mol0lrRBy3g)")

# File uploader
uploaded_file = st.file_uploader("Upload an MP4 File Containing a Person Walking.", type=["mp4"])

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
    RTOEy, LTOEy, fRTOEy, fLTOEy, fps, angles_data = process_video(input_video_path, output_video_path, output_video_path2)

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
    ax.plot(np.arange(0, len(fRTOEy)) / fps, fRTOEy, label="Filtered Data", color="red", marker='x')
    ax.set_title("Right Toe Y-Coordinate (RAW DATA)")
    ax.set_xlabel("Time (seconds, Progression of Gait)")
    ax.set_ylabel("Y-Coordinate")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # Finding Dominant Frequency (excluding 0Hz) Using (Positive) Magnitude Spectrum
    def find_dominant_frequency(data, sampling_freq):
        # Compute the Fourier Transform
        n = len(data)
        freq = np.fft.fftfreq(n, d=1/sampling_freq)  # Frequency bins
        fft_values = np.fft.fft(data)

        # Compute magnitude spectrum
        magnitude = np.abs(fft_values)

        # Exclude 0 Hz and nearby frequencies
        positive_freqs = freq[:n // 2]
        positive_magnitude = magnitude[:n // 2]

        # Ignore the first few bins near 0 Hz
        threshold_index = 1  ### Can adjust this if needed
        dominant_index = np.argmax(positive_magnitude[threshold_index:]) + threshold_index

        dominant_frequency = positive_freqs[dominant_index]
        return dominant_frequency

    critical_freq = max(0.3, find_dominant_frequency(fRTOEy, fps))
    # Apply Low Pass Filter through Fourier Transform to smoothen signal; max_value controlled by Nyquist Sampling Theorem
    st.write(f"Suggested Critical (Walking) Frequency: {critical_freq:.2f} Hz") # Suggested Optimal Critical Frequency by Gradient Descent
    with st.expander("Click here to adjust if the cadence below does not seem right!"):
        st.write('A normal walking frequency is typically below 2.0 Hz')
        critical_freq = st.slider("Adjust Critical Frequency such that distinct peaks can be observed, use the new graph below to gauge.", min_value=0.1, max_value=fps / 2, value=critical_freq, step=0.1)
        fRTOEy = fourier_filter(fRTOEy, fps, critical_freq) # Update fRTOEy data

        # Plot of filtered y-coordinates for the right toe.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(0, len(fRTOEy)) / fps, fRTOEy, label="Filtered Data", color="red", marker='x')
        ax.set_title("Right Toe Y-Coordinate (FILTERED DATA)")
        ax.set_xlabel("Time (seconds, Progression of Gait)")
        ax.set_ylabel("Y-Coordinate")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

    # Extract peaks
    peaks = extract_peaks(fRTOEy)
    peak_df = pd.DataFrame(peaks, columns=["Frame Index", "Y-Coordinate"])

    st.subheader("Select Range for Gait Analysis")
    st.write('Select range of the video where the **entire person can be seen from head to toe walking**.')
    # Time range selection
    time_values = np.arange(0, len(fRTOEy) / fps, 1 / fps)  # Times in seconds
    time_values = [f"{time:.2f} s (Frame {int(time * fps)})" for time in time_values] # With corresponding frame numbers
    time_min = st.selectbox("Select minimum time (seconds):", options=time_values, index=0)
    time_max = st.selectbox("Select maximum time (seconds):", options=time_values, index=len(time_values) - 1)
    frame_min = int(float(time_min.split(" ")[0]) * fps)
    frame_max = int(float(time_max.split(" ")[0]) * fps)
    if frame_min > frame_max:
        st.error("Minimum time must be less than or equal to maximum time.")
    else:
        st.write(f"Selected range: {time_min} to {time_max}, {round(100 * abs(frame_max - frame_min) / (len(fRTOEy) - 1),2)}% of the video is selected")

    st.subheader("Peak Detection")
    # Filter peaks based on selected frame range
    filtered_peaks = peak_df[(peak_df["Frame Index"] >= frame_min) & (peak_df["Frame Index"] <= frame_max)]

    # Display table of filtered peaks
    filtered_peaks = filtered_peaks.copy()
    filtered_peaks["Time (s)"] = filtered_peaks["Frame Index"] / fps
    columns_order = ["Time (s)"] + [col for col in filtered_peaks.columns if col != "Time (s)"]
    filtered_peaks = filtered_peaks[columns_order]
    if st.button('Click to display table of y-coordinate peaks for right toe'):
        st.write("Peaks within the selected range:")
        st.write(filtered_peaks)

    # Calculate gait characteristics if there are enough peaks
    if len(filtered_peaks) >= 2:
        # Calculate step length (difference between consecutive peaks)
        filtered_peaks['Step Length'] = filtered_peaks["Frame Index"].diff().fillna(0)

        # Calculate stride duration (difference between every second peak, assuming stride involves 2 steps)
        stride_durations = [
            filtered_peaks["Frame Index"].iloc[i + 1] - filtered_peaks["Frame Index"].iloc[i]
            for i in range(len(filtered_peaks) - 1)
        ]

        # Display calculated characteristics
        st.subheader("Gait Characteristics")
        st.write("Here are some Gait Characteristics calculated using data within the time frame you have selected.")
        step_duration = round((filtered_peaks['Step Length'].mean()) / 2, 2)
        st.write(f"**Average Step Duration (Right Leg)**: {round(step_duration/fps,3)} seconds") # step_duration frames
        cycle_time = round(np.mean(stride_durations), 2)
        st.write(f"**Average Cycle Time (Right Leg)**: {round(cycle_time/fps,3)} seconds") # cycle_time frames
        cadence = round(60 / (step_duration/fps),1) # Number of steps per minute
        st.write(f"**Cadence**: {cadence} steps / minute")
        num_of_steps = (len(fRTOEy)/fps) / ((filtered_peaks['Step Length'].mean()) / 2/fps)
        st.write(f"**Total Number of Steps**: {round(num_of_steps)} steps")
    else:
        st.error("Not enough peaks in the selected range to calculate gait characteristics.")

    # Joint Angles: Use duration between troughs of Hip Angles to determine duration of 1 Gait Cycle for angle plots.
    st.subheader('Joint Angles')
    st.write("Hover over the top right corner of the table:")
    st.write('Tip 1: Typing a negative sign(-) in the search box highlights all negative values.')
    st.write('Tip 2: The download button allows you to export the following data as a CSV file.')
    st.write(angles_data)

    # Ensure 'Hip Flexion-Extension (deg)' column is numeric
    angles_data['Hip Flexion-Extension (deg)'] = pd.to_numeric(
        angles_data['Hip Flexion-Extension (deg)'], errors='coerce'
    )
    angles_data = angles_data.dropna(subset=['Hip Flexion-Extension (deg)'])

    # Find critical frequency
    critical_freq = max(0.3, find_dominant_frequency(angles_data['Hip Flexion-Extension (deg)'], fps))

    # Apply Fourier filter
    angles_data['Hip Flexion-Extension (deg)'] = fourier_filter(
        angles_data['Hip Flexion-Extension (deg)'].values,  # Convert to NumPy array
        fps,
        critical_freq
    )

    # Extract troughs from filtered data
    sliced_data = angles_data['Hip Flexion-Extension (deg)'][frame_min:frame_max].reset_index(drop=True)
    troughs = extract_troughs(sliced_data)
    trough_df = pd.DataFrame(troughs, columns=["Frame Index", "Y-Coordinate"])

    # Function to normalize data for a single gait cycle
    def normalize_to_gait_cycle(data, troughs, num_points=100):
        cycles = []
        for i in range(len(troughs) - 1):
            start, end = troughs[i][0], troughs[i + 1][0]
            cycle_data = data[start:end]
            # Interpolate to fit `num_points` evenly spaced points
            normalized_cycle = np.interp(np.linspace(0, 1, num_points), np.linspace(0, 1, len(cycle_data)), cycle_data,)
            cycles.append(normalized_cycle)
        return np.array(cycles)

    # Hip angles: Normalize to percentage gait cycle
    normalized_cycles = normalize_to_gait_cycle(angles_data['Hip Flexion-Extension (deg)'], troughs)
    mean_cycle = np.mean(normalized_cycles, axis=0)
    std_cycle = np.std(normalized_cycles, axis=0)

    st.write("A 'Single Gait Cycle' is defined as 'Heel Strike' to 'Terminal Swing' below.")

    # Hip angles plot
    with st.expander("Click to Show Graph: Hip Flexion-Extension Across a Single Gait Cycle"):
        st.write("Note: Hip Extension: > 0°; Hip Flexion: < 0°")
        x = np.linspace(0, 100, len(mean_cycle))  # Percent gait cycle
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, mean_cycle, label="Mean Hip Angle", color="blue")
        ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("Hip Flexion-Extension (degrees)")
        ax.set_title("Hip Angle Progression Across a Single Gait Cycle")
        ax.legend()
        st.pyplot(fig)
        st.write("Note: From Heel Strike (0%) to Terminal Swing (100%)")

    # Knee angles: Normalize to percentage gait cycle
    normalized_cycles = normalize_to_gait_cycle(angles_data['Knee Flexion-Extension (deg)'], troughs)
    mean_cycle = np.mean(normalized_cycles, axis=0)
    std_cycle = np.std(normalized_cycles, axis=0)

    # Knee angles plot
    with st.expander("Click to Show Graph: Knee Flexion-Extension Across a Single Gait Cycle"):
        st.write("Note: Knee Extension: > 0°; Knee Flexion: < 0°")
        x = np.linspace(0, 100, len(mean_cycle))  # Percent gait cycle
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, mean_cycle, label="Mean Knee Angle", color="blue")
        ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("Knee Flexion-Extension (degrees)")
        ax.set_title("Knee Angle Progression Across a Single Gait Cycle")
        ax.legend()
        st.pyplot(fig)
        st.write("Note: From Heel Strike (0%) to Terminal Swing (100%)")

    # Ankle angles: Normalize to percentage gait cycle
    normalized_cycles = normalize_to_gait_cycle(angles_data['Ankle Dorsiflexion-Plantarflexion (deg)'], troughs)
    mean_cycle = np.mean(normalized_cycles, axis=0)
    std_cycle = np.std(normalized_cycles, axis=0)

    # Ankle angles plot
    with st.expander("Click to Show Graph: Ankle Dorsiflexion-Plantarflexion Across a Single Gait Cycle"):
        st.write("Note: Ankle Plantarflexion: > 0°; Ankle Dorsiflexion: < 0°")
        x = np.linspace(0, 100, len(mean_cycle))  # Percent gait cycle
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(x, mean_cycle, label="Mean Ankle Angle", color="blue")
        ax.fill_between(x, mean_cycle - std_cycle, mean_cycle + std_cycle, color="blue", alpha=0.2, label="±1 STD")
        ax.axhline(0, color="red", linestyle="--", linewidth=1.5, label="0°")
        ax.set_xlabel("% Gait Cycle")
        ax.set_ylabel("Ankle Dorsiflexion-Plantarflexion (degrees)")
        ax.set_title("Ankle Angle Progression Across a Single Gait Cycle")
        ax.legend()
        st.pyplot(fig)
        st.write("Note: From Heel Strike (0%) to Terminal Swing (100%)")

    ### GAIT SYMMETRY USING ROBINSON INDEX ###
    st.subheader("Gait Symmetry using Robinson Index")
    st.write("Note: Inputs in this section will cause the application to load.")
    default_dist = float(num_of_steps * 0.7)
    st.write(f"For a total of {round(num_of_steps)} steps:")
    distance_input = st.number_input("Total distance traveled by person in meters (Default value calculated using right leg step length of 70cm):",
                                     min_value = 0.0,
                                     value = default_dist,
                                     step = 0.1)

    st.write("To calculate symmetry, we need to track the movement of the **left toe** too!")
    critical_freq = max(0.3, find_dominant_frequency(fLTOEy, fps))
    # Apply Low Pass Filter through Fourier Transform to smoothen signal; max_value controlled by Nyquist Sampling Theorem
    st.write(f"Suggested Critical (Walking) Frequency (Left Foot): {critical_freq:.2f} Hz") # Suggested Optimal Critical Frequency by Gradient Descent
    with st.expander("Click to adjust for the left toe like what was done for the right toe above"):
        st.write('A normal walking frequency is typically below 2.0 Hz')
        critical_freq = st.slider("Adjust Critical Frequency such that distinct peaks can be observed, use the new graph below to gauge", min_value=0.1, max_value=fps / 2, value=critical_freq, step=0.1)
        fLTOEy = fourier_filter(fLTOEy, fps, critical_freq) # Update fRTOEy data

        # Plot of filtered y-coordinates for the left toe.
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(np.arange(0, len(fLTOEy)) / fps, fLTOEy, label="Filtered Data", color="red", marker='x')
        ax.set_title("Left Toe Y-Coordinate (FILTERED DATA)")
        ax.set_xlabel("Time (seconds, Progression of Gait)")
        ax.set_ylabel("Y-Coordinate")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)
    # Extract left peaks
    left_peaks = extract_peaks(fLTOEy)
    left_peak_df = pd.DataFrame(left_peaks, columns=["Frame Index", "Y-Coordinate"])
    # Filter left peaks based on selected frame range
    left_filtered_peaks = left_peak_df[(left_peak_df["Frame Index"] >= frame_min) & (left_peak_df["Frame Index"] <= frame_max)]

    # Display table of filtered peaks
    left_filtered_peaks = left_filtered_peaks.copy()
    left_filtered_peaks["Time (s)"] = left_filtered_peaks["Frame Index"] / fps
    columns_order = ["Time (s)"] + [col for col in left_filtered_peaks.columns if col != "Time (s)"]
    left_filtered_peaks = left_filtered_peaks[columns_order]
    if st.button('Click to display table of y-coordinate peaks for left toe'):
        st.write("Left Peaks within the selected range:")
        st.write(left_filtered_peaks)

    # Calculate gait characteristics if there are enough left peaks
    if len(left_filtered_peaks) >= 2:
        # Calculate step length (difference between consecutive left peaks)
        left_filtered_peaks['Step Length'] = left_filtered_peaks["Frame Index"].diff().fillna(0)

        # Calculate stride duration (difference between every second left peak, assuming stride involves 2 steps)
        left_stride_durations = [
            left_filtered_peaks["Frame Index"].iloc[i + 1] - left_filtered_peaks["Frame Index"].iloc[i]
            for i in range(len(left_filtered_peaks) - 1)
        ]

        # Display calculated characteristics
        st.write("Gait Characteristics for the left toe:")
        left_step_duration = round((left_filtered_peaks['Step Length'].mean()) / 2, 2)
        st.write(f"**Average Step Duration (Left Leg)**: {round(step_duration/fps,3)} seconds") # step_duration frames
        left_cycle_time = round(np.mean(left_stride_durations), 2)
        st.write(f"**Average Cycle Time (Left Leg)**: {round(cycle_time/fps,3)} seconds") # cycle_time frames
    else:
        st.error("Not enough peaks in the selected range to calculate gait characteristics.")

    # Checkbox for the user to select the affected leg
    st.write("Select the Affected Leg")
    affected_leg = st.radio(
        "Choose which leg is affected:",
        options=["Left Leg", "Right Leg"]
    )

    # Define parameters for affected and unaffected legs
    Xa_label = "Affected Left Leg Step Length" if affected_leg == "Left Leg" else "Right Leg Step Length"
    Xu_label = "Affected Right Leg Step Length" if affected_leg == "Left Leg" else "Left Leg Step Length"

    # Calculate step length for both legs
    if cadence > 0 and distance_input > 0:
        left_step_length = (left_step_duration/fps / ((len(LTOEy)/fps))) * distance_input
        right_step_length = distance_input / num_of_steps
        st.write(f"**Left Leg Step Length:** {round(left_step_length,5)} meters")
        st.write(f"**Right Leg Step Length:** {round(right_step_length,5)} meters")

        # Assign Xa and Xu based on affected leg
        Xa = left_step_length if affected_leg == "Left Leg" else right_step_length
        Xu = right_step_length if affected_leg == "Left Leg" else left_step_length

        # Calculate Robinson Index (SI)
        SI = 100 * (Xa - Xu) / Xu

        # Display results
        st.write("""
The **Robinson Index (SI)** is a measure of gait asymmetry, specifically focusing on the step length of the affected and unaffected legs. (Formula Reference: Robinson et al., 1987) 
- \( Xa \): Step length of the **affected leg** (in meters).
- \( Xu \): Step length of the **unaffected leg** (in meters).

**Significance of SI**
- **\( SI = 0 \):** Perfect symmetry; both legs have the same step length.
- **\( SI > 0 \):** The affected leg has a longer step length than the unaffected leg (e.g. overcompensation).
- **\( SI < 0 \):** The affected leg has a shorter step length than the unaffected leg (e.g. injury or reduced mobility).
- **Magnitude of SI:** Larger values indicate greater asymmetry and more pronounced gait imbalance. Values close to 0 indicate normal variations.
""")
        st.latex(r"Robinson~Index~SI = 100 \times \frac{X_a - X_u}{X_u}")
        st.write(f"## **Robinson Index (SI):** {SI:.2f}")

    else:
        st.warning("Please provide a valid distance traveled and ensure cadence is available.")

    ### COMBINED EXPORT TO CSV ###
    st.title("📊 EXPORT COMBINED DATA")
    st.markdown("""
    You can download a CSV containing the following information:
    - **Frame Index**
    - **Time**
    - **Y Coordinates** for movement of the left toe (Raw data)
    - **Y Coordinates** for movement of the left toe (Noise corrected)
    - **Y Coordinates** for movement of the right toe (Raw data)
    - **Y Coordinates** for movement of the right toe (Noise corrected)
    - **Right Hip Flexion-Extension Angle**
    - **Right Knee Flexion-Extension Angle**
    - **Right Knee Flexion-Extension Angle**

    Click the button below to download the file.
    """)
    combined_data = pd.DataFrame({
        "Frame Index": np.arange(len(RTOEy)),
        "Time (s)": np.arange(len(RTOEy)) / fps,
        "LTOEy (Raw)": LTOEy,
        "LTOEy (Noise Corrected)": fLTOEy,
        "RTOEy (Raw)": RTOEy,
        "RTOEy (Noise Corrected)": fRTOEy,
        "Right Hip Flexion-Extension Angle (deg)": angles_data["Hip Flexion-Extension (deg)"].values,
        "Right Knee Flexion-Extension Angle (deg)": angles_data["Knee Flexion-Extension (deg)"].values,
        "Right Ankle Dorsiflexion-Plantarflexion Angle (deg)": angles_data["Ankle Dorsiflexion-Plantarflexion (deg)"].values,
    })
    csv_data = combined_data.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Click to Export All Data to CSV",
        data=csv_data,
        file_name="gait_analysis_data.csv",
        mime="text/csv"
    )

# Instructions to Refresh the page
st.info('Refresh the Page or click on \'Browse Files\' again to process the next video.')
st.warning('Note: Download your processed videos before refreshing if you want to keep them.')
