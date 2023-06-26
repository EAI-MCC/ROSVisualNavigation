import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API

print("Environment Ready")

# Setup:
pipe = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipe)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Pass the config
pipe.start(config)

# Grab colorizer object
colorizer = rs.colorizer()

# Skip 5 first frames to give the Auto-Exposure time to adjust
for x in range(5):
    pipe.wait_for_frames()
    
# Used for trackbar calls
def nothing(x):
    pass


cv2.namedWindow("Tracking")
cv2.createTrackbar("LH", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LS", "Tracking", 0, 255, nothing)
cv2.createTrackbar("LV", "Tracking", 0, 255, nothing)
cv2.createTrackbar("UH", "Tracking", 255, 255, nothing)
cv2.createTrackbar("US", "Tracking", 255, 255, nothing)
cv2.createTrackbar("UV", "Tracking", 255, 255, nothing)

# Write some Text
font = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10, 500)
fontScale = 1
fontColor = (255, 255, 255)
lineType = 2

while True:

    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    color_image = np.asanyarray(color_frame.get_data())

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("LH", "Tracking")
    l_s = cv2.getTrackbarPos("LS", "Tracking")
    l_v = cv2.getTrackbarPos("LV", "Tracking")

    u_h = cv2.getTrackbarPos("UH", "Tracking")
    u_s = cv2.getTrackbarPos("US", "Tracking")
    u_v = cv2.getTrackbarPos("UV", "Tracking")

    l_b = np.array([l_h, l_s, l_v])
    u_b = np.array([u_h, u_s, u_v])

    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(hsv, l_b, u_b)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    contours = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[-2]

    # targets
    for c in contours:
        # Draw contours
        cv2.drawContours(color_image, contours, -1, (0, 255, 0), 3)

        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and centroid thingy
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        cv2.circle(color_image, center, 5, (0, 0, 255), -1)

    # Create alignment primitive with color as its target stream:
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    # Update color and depth frames:
    aligned_depth_frame = frameset.get_depth_frame()  # IMPORTANT!!
    aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

    colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())

    distance = aligned_depth_image[center[1], center[0]]

    cv2.putText(color_image, f"{distance} mm",
                (center[0] + 10, center[1] - 5),
                font,
                fontScale,
                fontColor,
                lineType)

    cv2.imshow("RGB Image", color_image)
    cv2.imshow("Depth Image", colorized_depth)

    cv2.waitKey(1)

    # detect keys
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key != 255:
        print('KEY PRESS:', [chr(key)])

# Cleanup:
pipe.stop()
print("Frames Complete")