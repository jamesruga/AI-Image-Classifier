Developing an AI Application

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, you might want to include an image classifier in a smart phone app. To do this, you'd use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications.

In this project, I'll train an image classifier to recognize different species of flowers. You can imagine using something like this in a phone app that tells you the name of the flower your camera is looking at. In practice you'd train this classifier, then export it for use in your application. We'll be using a dataset of 102 flower categories, you can see a few examples below.



The project is broken down into multiple steps:

 Load and preprocess the image dataset

 Train the image classifier on your dataset

 Use the trained classifier to predict image content

 We'll lead you through each part which you'll implement in Python.

You'll have an application that can be trained on any set of labeled images. Here our network will be learning about flowers and end up as a command line application. But, this new app can be used with another dataset depending on your imagination and effort in building a dataset. For example, imagine an app where you take a picture of a car, it tells you what the make and model is, then looks up information about it. Go build your own dataset and make something new.