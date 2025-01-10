import unittest
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class WebAppTest(unittest.TestCase):
    results = []

    def setUp(self):
        self.driver = webdriver.Chrome()  
        self.base_url = "http://127.0.0.1:5000"  # URL of the Flask application

    def test_image_upload_feature(self):
        driver = self.driver
        driver.get(self.base_url)

        try:
            # Wait for the file input element to be visible and interactable
            file_input = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.ID, "file"))
            )
            file_input.send_keys("/Users/armeenmobasher/Desktop/DSC3/Project/Dataset/normal/normal1.png")  # Correct path to an image file

            # Find the submit button by ID and click it
            submit_button = WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.ID, "submit"))
            )
            submit_button.click()

            # Wait for the page to indicate the upload has been successful
            WebDriverWait(driver, 20).until(
                EC.text_to_be_present_in_element((By.TAG_NAME, "body"), "Predicted Class: Normal")
            )
            # Confirmation message check
            page_text = driver.find_element(By.TAG_NAME, "body").text
            self.assertIn("Predicted Class: Normal", page_text)
            self.results.append(("Image Upload Feature", "✓ Passed"))
        except Exception as e:
            print(f"Error during test: {str(e)}")
            self.results.append(("Image Upload Feature", "✗ Failed"))

    def tearDown(self):
        self.driver.quit()

    @classmethod
    def tearDownClass(cls):
        print("\nTest Results Summary:")
        print("| Feature                   | Result    |")
        print("|---------------------------|-----------|")
        for test, result in cls.results:
            print(f"| {test.ljust(25)} | {result.ljust(9)} |")

if __name__ == "__main__":
    unittest.main()
