import time
import pynput
import ctypes
import wx
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Key, Controller as KeyboardController

width, height = 1280, 720

# calibrate mouse settings - sensitivity sets how much larger the source frame is relative to the screen size
def mouseCalibration(width, height, sensitivity = 2.5):
	# get screen size - works for windows, not tested on other platforms
	screen_wx = wx.App(False)
	screen_width, screen_height = wx.GetDisplaySize()

	# mouse_scale is how much to multiply the width, mouse_offset is to ensure the screen and mouse-dimension are centered
	mouse_scale = max((screen_width * sensitivity) / width, 1)
	mouse_offset = (max((mouse_scale * width - screen_width) / 2, 0), max((mouse_scale * height - screen_height) / 2, 0))
	return mouse_scale, mouse_offset

# fix certain Windows related mouse issues
PROCESS_PER_MONITOR_DPI_AWARE = 2
ctypes.windll.shcore.SetProcessDpiAwareness(PROCESS_PER_MONITOR_DPI_AWARE)

# main function to convert gestures into functions
def gestureToFunction(q, width, height):
	mouse_scale, mouse_offset = mouseCalibration(width, height)

	# instantiate a virtual keyboard/mouse
	keyboard = KeyboardController()
	mouse = MouseController()
	t = time.time()
	wait = time.time()
	interval = 1.0
	block = False
	# dictionary of mouse-related variables
	mouseData = {
		"old_pos": (0,0),
		"real_pos": mouse.position,
		"direction": (0,0),
		"allow_click": True,
		"allow_rightclick": True,
		"allow_scroll": True,
		"allow_click": True
	}
	previous_mouse_pos = (0,0)
	while True:
		gesture = q.get()

		# scrollup part
		if gesture["current"] == "peaceup":
			mouseData["allow_scrollup"] = True

		if gesture["previous"] == "peaceup" and gesture["current"] == "peacedown" and time.time() > wait and mouseData["allow_scrollup"]:
			wait = time.time() + interval
			mouse.scroll(0, 5000)
			mouseData["allow_scrollup"] = False
			print("scroll up")

		# scrolldown part
		if gesture["current"] == "peacedown":
			mouseData["allow_scrolldown"] = True

		if gesture["previous"] == "peacedown" and gesture["current"] == "peaceup" and time.time() > wait and mouseData["allow_scrolldown"]:
			wait = time.time() + interval
			mouse.scroll(0, -5000)
			mouseData["allow_scrolldown"] = False
			print("scroll down")

		# cursor-movement part
		if gesture["current"] == "point" or (gesture["previous"] == "point" and (gesture["current"] == "pinch" or gesture["current"] == "ok")):
			target_position = (gesture["mouse_position"][0] * mouse_scale - mouse_offset[0], gesture["mouse_position"][1] * mouse_scale - mouse_offset[1])
			direction = (target_position[0] - mouse.position[0], target_position[1] - mouse.position[1])
			direction = unit(direction[0], direction[1])
			new_position = (mouse.position[0] + direction[0], mouse.position[1] + direction[1])
			if target_position[0] in range(int(mouse.position[0]), int(new_position[0])):
				new_position = (target_position[0], new_position[1])
			if target_position[1] in range(int(mouse.position[1]), int(new_position[1])):
				new_position = (new_position[0], target_position[1])
			mouse.position = new_position

		# left-click part
		if gesture["current"] == "pinch":
			mouseData["allow_click"] = True

		if gesture["previous"] == "pinch" and mouseData["allow_click"]:
			mouse.click(Button.left, 1)
			mouseData["allow_click"] = False
			print("leftclick")

		# right-click part
		if gesture["current"] == "ok":
			mouseData["allow_rightclick"] = True

		if gesture["previous"] == "ok" and mouseData["allow_rightclick"]:
			mouse.click(Button.right, 1)
			mouseData["allow_rightclick"] = False
			print("rightclick")

# convert velocity into speed, for added smoothness of mouse movement
def unit(dir_x, dir_y):
	scalar = (dir_x**2 + dir_y**2)**0.5
	return (dir_x/5, dir_y/5)