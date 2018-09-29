# alibaba-hackathon-
reviews classification 
Zhejiang Lab & Alibaba Cloud Tianchi

Natural Language Processing Hackathon @ BML MUNJAL UNIVERSITY, India

 

1. Topic

One of the widely used natural language processing task is “Text Classification”. The goal of the text classification is to automatically classify the text documents into one or more defined categories, such as understanding audience sentiment from social media, detection of spam of non-spam emails etc. This competition requires participants to perform binary classification on each short text, i.e. predict its label as “_label_1” or “_label_2”.

 

2. Data Instruction

Each training sample in Dataset A has the “label text” format, e.g.

_label_1 Angry: I bought this bed while I decided to purchase a normal bed. I was at first excited but as the week, past the seam popped and the bed now has a bubble and un-usable. for the money you pay for an air-bed I stick with a traditional bed over and air bed. Least to say thanks the quality is poor! I do not recommend.

 

Each testing sample in Dataset B has only the text description, e.g.

Cheap price good product: My parents use this and it's made by motorola. No issues with it what so ever

 

For each testing sample, the participants should predict whether its label is _label_1 or _label_2.

 

3. Submission Instruction

The competition requires participants to submit the predicted labels in the order in which the testing texts are offered. Each line in the submitted txt file contains only one predicted label and can’t contain other characters.

 

4. Grading standard

We use the average F1 score to measure the performance of text classification.

F1 score is the harmonic average of the precision and recall, which reaches its best value at 1 and worst at 0.

The precision is the number of true positives (i.e. the number of samples correctly labeled as belonging to the positive class) divided by the total number of samples labeled as belonging to the positive class.

The recall is defined as the number of true positives divided by the total number of samples that actually belong to the positive class.

 

 

5. Attention

This competition does not prohibit participants from using algorithms from other places, but it will strictly prohibit participants from using any external data to train models. The organizer of this competition has the right to check the participants’ codes (especially the top three teams). If the above rule is violated, the ranking will be cancelled.

 

6. Announcement

(1) This competition’s primary data source is from Internet.

(2) If participants’ personal interests have been infringed, please contact competition committee, whose will reply to participants as soon as possible.
