import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description="Histogram-based price calculator for printing")
parser.add_argument("-i", "--image", required=True, help="Path to input image")
args = vars(parser.parse_args())

img = cv2.imread(args["image"])
if img is None:
    raise SystemExit("Error: Image not found")
img = cv2.resize(img, (600, 900))
cv2.imshow("Original Page", img)

height, width = img.shape[:2]
quadrants = {
    "Top Left (Q1)": img[0:height//2, 0:width//2],
    "Bottom Left (Q2)": img[height//2:height, 0:width//2],
    "Top Right (Q3)": img[0:height//2, width//2:width],
    "Bottom Right (Q4)": img[height//2:height, width//2:width]
}

print("-----------------------------------------------")
print("             Price Calculator                  ")
print("-----------------------------------------------")

def compute_price(section, label, fig_name):
    chans = cv2.split(section)
    white_est = 0

    plt.figure(num=fig_name)
    plt.title(f"Histogram - {label}")
    plt.xlabel("Bins")
    plt.ylabel("Pixel Count")

    for chan, c in zip(chans, ("b", "g", "r")):
        hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
        white_est += hist[230:256].sum()  # estimate near-white
        plt.plot(hist, color=c)
        plt.xlim([0, 256])

    total = section.shape[0] * section.shape[1]
    avg_white = white_est / 3.0
    colored = total - avg_white
    ratio = colored / total  

    contrib = 5.0 * ratio   
    contrib = round(contrib, 2)

    print(f"{label} - Price Contribution = ₱{contrib:.2f}")
    return contrib

contributions = []
for idx, (name, q) in enumerate(quadrants.items(), start=1):
    contrib = compute_price(q, name, f"Q{idx} Histogram")
    contributions.append(contrib)

total_price = sum(contributions)
final_price = round(min(max(total_price, 2.0), 20.0), 2)

print("______________________________________________")
print(f"\nPAGE PRICE = ₱{final_price:.2f}")


plt.figure(num="Full Page Histogram")
plt.title("Flattened Color Histogram (Full Page)")
plt.xlabel("Bins")
plt.ylabel("Pixel Count")
for chan, c in zip(cv2.split(img), ("b", "g", "r")):
    hist = cv2.calcHist([chan], [0], None, [256], [0, 256])
    plt.plot(hist, color=c)
    plt.xlim([0, 256])

plt.show()
cv2.waitKey(0)
cv2.destroyAllWindows()
