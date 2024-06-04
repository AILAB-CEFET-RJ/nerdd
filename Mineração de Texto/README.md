# HANDY (MÃOZINHA) - Offensive Gesture Detector 👋🚫 

**HANDY (MÃOZINHA)** is a tool that utilizes the pretrained **MediaPipe** model to detect the presence of a hand in an image or video. Additionally, it employs another pretrained model (which will be mentioned later) to perform classification and check for an offensive gesture, specifically the middle finger gesture.

This project is a fun and interesting way to explore the power of computer vision and real-time gesture detection. Let's help you set up and run HANDY in your environment.

## Prerequisites 📋

Make sure you have the following prerequisites installed in your development environment:

- Python 3.x
- Pip (Python package manager)

## How to Run 🏃‍♀️

Follow these simple steps to set up and run HANDY:

1. **Clone the Repository**

   ```shell
   git clone https://github.com/birdred/handy.git
   ```

2. **Install Dependencies**

   ```shell
   cd handy
   pip install -r requirements.txt
   ```

3. **Run HANDY**

   ```shell
   python main.py
   ```

   Now you should see the live camera window with hand detection and offensive gesture classification.

## Test Codes 🧪

In the `tests/` directory, you will find some sample images and videos to test HANDY's functionality. To run the tests, you can use the following command:

```shell
python test.py
```

This will run the offensive gesture detector on the test images and videos and display the results in the output.

## Contribution 🤝

If you want to contribute to the HANDY project, we would be happy to receive your contributions. Feel free to open issues or submit pull requests with improvements, bug fixes, or new features.

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

We hope you enjoy offensive gesture detection with **HANDY (MÃOZINHA)**! If you have any questions or need assistance, please feel free to reach out to the development team. 😊👋
