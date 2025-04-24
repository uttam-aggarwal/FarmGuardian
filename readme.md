# FarmGuardian 🌾🐄

**FarmGuardian** is an intelligent livestock monitoring system designed to help farmers manage and protect their livestock by providing real-time tracking and alerting. With the power of YOLOv8 object detection and live camera feeds, it ensures that no animal is left unmonitored, alerting users to abnormal behavior and tracking animal movements efficiently.

---

## 🌟 Features

- **Real-time Animal Tracking**: Detects and tracks animals such as birds, cats, dogs, and farm livestock (sheep, cows, horses, etc.) in live camera feeds.
- **Customizable Camera Zones**: Set up multiple cameras across different farm zones (e.g., North Pasture, South Barn) for enhanced monitoring.
- **Instant Alerts**: Receive alerts when a specific animal is detected in a zone based on configurable thresholds for consecutive frames and cooldown periods.
- **Animal Movement Trail**: Visualize the movement path of animals across the screen, helping you monitor behavior patterns.
- **Database Logging**: All activities and alerts are logged in an SQLite database, providing easy access to historical data.
- **Grid Display**: View live camera feeds in a customizable grid layout for efficient surveillance.
- **Offline Camera Handling**: Automatically handles camera feed interruptions by marking offline cameras with a placeholder.

---

## 🔧 Technologies Used

- **Python**: Main programming language
- **YOLOv8**: For real-time object detection
- **OpenCV**: For camera feed processing and visualization
- **SQLite**: For database management
- **Threading**: For simultaneous processing of multiple camera feeds

---

## 🖥️ Setup and Installation

1. **Clone the Repository**:

    ```bash
    git clone https://github.com/yourusername/farmguardian.git
    cd farmguardian
    ```

2. **Install Dependencies**:

    Use pip to install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

    Ensure you have `opencv`, `yolo`, and other necessary libraries installed.

3. **Configure Your Cameras**:

    Modify the `CONFIG` dictionary to add your camera sources and desired configurations.

4. **Run the System**:

    Execute the script to start monitoring:

    ```bash
    python main.py
    ```

---

## 🚨 Alerts and Notifications

FarmGuardian triggers alerts based on configurable conditions:
- **Consecutive Frames**: The number of consecutive frames an animal must appear before triggering an alert.
- **Cooldown Period**: Minimum time between two consecutive alerts for the same animal.
- **Activity Window**: The time window in which active animals are detected.

---


All alerts and animal activities are logged in a database (`farm_logs.db`). You can query the last 5 alerts directly from the system:

l -> View last 5 alerts

## 🎨 Customization
You can adjust the following configurations in the CONFIG dictionary to tailor the system to your farm:

**Farm and Camera Zones:** Specify different zones and cameras for coverage.

**Model:** Customize the YOLOv8 model and detection parameters.

**Alert Settings:** Fine-tune the alert thresholds and activity windows to match your needs.

**System Layout:** Choose between different grid layouts and camera resolutions.

## 📸 Example Output

Alert Notification Example:
```bash
[ALERT] Thu, 29 Apr 2025 12:35:24
- Triggered by: DOG (ID: 12)
- Location: South Barn
- Active animals: 2 SHEEP(s) in North Pasture, 1 CAT(s) in South Barn
- Total count: 3
```
## 🛠️ Future Improvements
Integration with IoT Devices: Integrate with smart sensors for enhanced farm management.


**Cloud Storage:** Store logs and alerts in the cloud for remote access.

**Mobile App:** Develop a mobile version of FarmGuardian for on-the-go monitoring.

## 🤝 Contributing
Feel free to fork the repository and submit pull requests. If you have any ideas or suggestions for improvements, please open an issue!

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.

## 💬 Contact
For any questions, suggestions, or inquiries, feel free to reach out to:

Uttam

Email: uttamaggarwal321@gmail.com