from pytube import YouTube 

link = input('https://youtu.be/AA7i2GcTGwU')
yt = YouTube(link)
#extract data from youtube_video
# To print title
print("Title :", yt.title)
# To get number of views
print("Views :", yt.views)
# To get the length of video
print("Duration :", yt.length)
# To get description
print("Description :", yt.description)
# To get ratings
print("Ratings :", yt.rating)

#download youtube_video
stream = yt.streams.get_highest_resolution()
stream.download()
print("Download Complete!!")