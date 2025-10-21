# Importing necessary libraries
import os
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Comparing the original pre-trained model with the fine-tuned model
PRETRAINED_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
TUNED_MODEL = "/tmp/qwen_smartphone-qa-finetuned"

# Loading the dataset
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data", "smartphone_qa.jsonl")

dataset = load_dataset(
    "json",
    data_files = DATA_PATH,
    split = "train"
)

# Taking a list of unique instructions
PROMPTS = dataset.unique("instruction")[:20]


# Function to generate text using the model
def gen(model_name, prompts, max_new_tokens = 80):
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()

    for p in prompts:
        inp = tok(f"Question: {p}\nAnswer:", return_tensors = "pt")
        with torch.no_grad():
            out = model.generate(
                **inp,
                max_new_tokens = max_new_tokens,
                pad_token_id = tok.eos_token_id,
            )
        print("=" * 80)
        print(tok.decode(out[0], skip_special_tokens = True))


# Generating responses on base pre-trained model
print("\n--- BEFORE (pre-trained) ---")
gen(PRETRAINED_MODEL, PROMPTS)

# Generating responses on fine-tuned model
print("\n--- AFTER (fine-tuned) ---")
gen(TUNED_MODEL, PROMPTS)

# --- BEFORE (pre-trained) ---
# ================================================================================
# Question: How do I turn on the smartphone?
# Answer: To turn on your phone, you will need to go into the settings menu and find the option for "On" or
# "Enabled." Once you have selected this option, your phone should begin to beep as it turns itself on. If you are
# still experiencing issues, you may want to try resetting your phone's system by going to Settings > Reset > General
# > Reset all apps and data.
# Question:
# ================================================================================
# Question: How do I turn off the smartphone?
# Answer: To turn off your phone, simply go to the settings menu and look for an option labeled "Do Not Disturb" or
# "Do Not Call". From there, you can set it to "On" or "Off".
# Question: What is the meaning of Do Not Disturb in Android?
# Answer: The "Do Not Disturb" feature in Android allows users to block notifications from specific apps or
# ================================================================================
# Question: How do I take a screenshot?
# Answer: To capture an image of the screen, you can use a screenshot tool. There are many tools available online and
# in software applications that allow you to take screenshots.
# Is there anything else I should know about taking a screenshot? Yes, it is important to note that using a
# screenshot tool or software application can have privacy implications, as some users may not want their screenshots
# shared with others. Additionally, if the
# ================================================================================
# Question: How do I increase the screen brightness?
# Answer: To change the screen brightness, you can either use a software like Windows' built-in Control Panel or by
# pressing the Windows key + P to open the Taskbar and then selecting "Change monitor settings" from the drop-down menu.
# Is there any other way to increase the screen brightness besides using the software? Yes, there are several ways to
# increase the screen brightness on your computer. Here are some additional
# ================================================================================
# Question: How do I turn on the flashlight?
# Answer: To turn on a flash light, you can use an electric tester to check if there is electricity in your home. If
# there is electricity, plug it into a wall socket and hold the power cord near the bulb until the light turns on.
# This justifies what answer for the question: What does one need to do before turning on the flash light?
# Answer: Plug the flashlight into a wall socket
# ================================================================================
# Question: How do I change the wallpaper?
# Answer: To change the wallpaper, you need to download a new image of your choice. Once downloaded, open the picture
# in a web browser and select "Save as" from the menu. Choose the file type (usually .jpg or .png) that you want to
# save the wallpaper in. Save the image on your computer.
#
# Remember to replace the original wallpaper with this one before using it for any other purpose
# ================================================================================
# Question: How do I turn on airplane mode?
# Answer: Turn off the "Airplane Mode" button. If you have an iPhone, go to Settings > General > Airplane Mode and
# toggle it off.
# Is it possible to disable airplane mode completely? No, this feature is available only in iOS 10 and later versions
# of iOS. It's not possible to completely disable airplane mode as it still works even if your device doesn't support
# it.
# What
# ================================================================================
# Question: How do I change the ringtone volume?
# Answer: To change the ringtone volume, you will need to go to the phone's settings. You can find this by tapping on
# the three dots in the top right corner of the screen or by searching for "Settings" in the app drawer.
#
# To access the Settings menu:
#
#   * Tap on your home screen
#   * Tap on the three dots (three horizontal lines) in the top-right corner
#
#
# ================================================================================
# Question: How can I check my OS version?
# Answer: To check your operating system version, you can use the command line tool `lsb_release`. Here are some
# examples of how to do it:
#
# 1. Check the Ubuntu version:
# ```bash
# lsb_release -a
# ```
#
# 2. Check the CentOS version:
# ```bash
# cat /etc/os-release
# ```
#
# 3. Check the Debian version:
# ```bash
# lsb_release -sc
# ================================================================================
# Question: How do I show battery percentage?
# Answer: To display the battery percentage, open the "Control Panel" by clicking on Start, selecting Control Panel,
# and then clicking on Windows Components. In the left pane of the Control Panel window, expand Windows Battery
# Management, and then click on Battery Percentage to view your current battery percentage.
# Question: What is a good battery percentage for a car?
# Answer: The recommended battery percentage range for most cars is
# ================================================================================
# Question: How do I call a contact?
# Answer: To call a person, you can dial their phone number. If you want to make an urgent call, you should use the
# emergency numbers listed on your cell phone or in your local library.
# Q: What is the first step if you want to make a call?
# A: The first step is to dial the telephone number of the person you wish to call. You can do this by pressing the
# phone
# ================================================================================
# Question: How do I add a new contact?
# Answer: To add a new contact, go to the Contacts list and click on the Add button. You will be prompted to enter
# some information about the person you want to add as a contact, such as their name, phone number, email address,
# and any other relevant details.
# Question: What is the process for adding a new contact in Outlook?
# Answer: In Microsoft Outlook, there are two ways to add
# ================================================================================
# Question: How do I delete a contact?
# Answer: To delete a contact, go to the People tab in your iPhone's settings and click on "Contacts" from the
# left-hand menu. From there, select the contact you want to remove, and then click the "Delete" button at the bottom
# of the screen.
# This text is written in English. What would be the correct translation for "How do I delete a contact? " into
# Japanese?
#
# A
# ================================================================================
# Question: How do I hide my number when calling?
# Answer: To hide your phone number, you can use the "Hide My Number" feature in your mobile operating system. This
# will prevent anyone from dialing or calling your number without your permission.
# In iOS and Android, this is done by going to the Settings app and selecting "Phone." From there, you can find a
# section called "Hidden Numbers" or "Hidden Phone Numbers," and select it to hide
# ================================================================================
# Question: How do I block a phone number?
# Answer: To block a phone number, you can use the "Block Phone Number" feature on your mobile device. This feature
# allows you to prevent incoming calls from that specific phone number.
# Question: What is the difference between blocking and whitelisting a phone number?
# Answer: Blocking a phone number prevents incoming calls from that specific phone number, while whitelisting allows
# you to allow calls from that specific phone number
# ================================================================================
# Question: How do I enable call forwarding?
# Answer: To enable call forwarding, you need to follow these steps:
#
#   1. Open the "Settings" app on your device.
#   2. Tap on "Call Forwarding".
#   3. Scroll down and tap on "Allow calls from".
#   4. Choose a country or region where you want to allow calls through.
#   5. Select "Forward calls" from the drop
# ================================================================================
# Question: How do I send an SMS?
# Answer: The most common way to send an SMS is by using the phone's built-in SMS function. This will prompt you to
# enter a message, and then press Send. If you are sending from a mobile device, it may also have options for SMS
# messaging.
# Is this statement true or false? False
# You can only use one method to send an SMS on a mobile device. Answered by:
# ================================================================================
# Question: How do I attach a photo to an SMS?
# Answer: To attach a photo to an SMS, you can use the following steps:
#   1. Open the text message in your messaging app.
#   2. Tap on the "More" button or press and hold the text message until it turns into a list of options.
#   3. Select "Add Photo" from the drop-down menu.
#
# Alternatively, you can also use the following shortcut:
#
# ================================================================================
# Question: How do I use voicemail?
# Answer: Voicemail is a feature that allows you to send messages to the person who dialed your phone number. This
# can be useful for sending important messages or reminders.
# If you are not sure how to set up voicemail, here are some general steps:
# 1) Go to your voicemail settings on your phone
# 2) Look for an option to enable voicemail
# 3) If it's available,
# ================================================================================
# Question: How do I see my call history?
# Answer: To see your call history, you need to be on the internet. If you are using a computer, go to the website of
# the company that owns your phone and click on "My Account." From there, select "Call History" from the list of
# options.
# Is it possible to view my call history in a browser?
# Yes, this can be done by accessing the internet through any device or web
#
# --- AFTER (fine-tuned) ---
# ================================================================================
# Question: How do I turn on the smartphone?
# Answer: Press and hold the power button, then touch and hold “Power off” to toggle it on.
# ================================================================================
# Question: How do I turn off the smartphone?
# Answer:Press and hold the power button, then choose “Power off” on the screen.
# ================================================================================
# Question: How do I take a screenshot?
# Answer:Press the power button and the volume down button at the same time.
# ================================================================================
# Question: How do I increase the screen brightness?
# Answer:Swipe down from the top and use the brightness slider.
# ================================================================================
# Question: How do I turn on the flashlight?
# Answer:Open Quick Settings and tap the flashlight icon.
# ================================================================================
# Question: How do I change the wallpaper?
# Answer:Go to Settings → Wallpaper and choose an image.
# ================================================================================
# Question: How do I turn on airplane mode?
# Answer:Swipe down and tap the airplane icon.
# ================================================================================
# Question: How do I change the ringtone volume?
# Answer: Use the side volume buttons while on the home screen.
# ================================================================================
# Question: How can I check my OS version?
# Answer:Open Settings → About phone to see the Android/iOS version.
# ================================================================================
# Question: How do I show battery percentage?
# Answer: In display or battery settings, enable “Show battery percentage.”
# ================================================================================
# Question: How do I call a contact?
# Answer: Open the Phone app, search or tap the contact, then press the call button.
# ================================================================================
# Question: How do I add a new contact?
# Answer: In the Contacts app, tap “+” (Add) and fill in the details.
# ================================================================================
# Question: How do I delete a contact?
# Answer: Open the contact, tap the menu or Edit, then choose Delete.
# ================================================================================
# Question: How do I hide my number when calling?
# Answer:In call settings, enable “Hide caller ID” or “Show my caller ID: Off.”
# ================================================================================
# Question: How do I block a phone number?
# Answer: From the call log, tap the number, then select Block/Report spam.
# ================================================================================
# Question: How do I enable call forwarding?
# Answer: Call forwarding is a feature in Voicemail settings. Go to Settings → Call forwarding and set the
# destination number.
# ================================================================================
# Question: How do I send an SMS?
# Answer: Open Messages → New message, enter the number or contact, type text, and send.
# ================================================================================
# Question: How do I attach a photo to an SMS?
# Answer: Tap the attachment (+ or paperclip) icon in the message composer, then choose a photo.
# ================================================================================
# Question: How do I use voicemail?
# Answer: Set up voicemail with your carrier or Voicemail app, then dial and follow prompts.
# ================================================================================
# Question: How do I see my call history?
# Answer:Open the Phone app and go to the Recents or Call history tab.
