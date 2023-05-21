# Deprecation Notice: birds-200-sagemaker-devops

**Note: This repository has been deprecated and is no longer actively maintained.**

The goal of this project was to create a system that could identify different bird species using AWS SageMaker. I aimed to train and deploy the system through a pipeline, with a web interface allowing users to upload bird images for detection, displaying bounding boxes and labels as results.

However, after careful consideration, I have decided to deprecate this repository. There are several reasons behind this decision. Firstly, the cost involved in training image-based models using AWS SageMaker is significant, and I have determined that the resources required for this toy project are not justified. Additionally, I found that the vanilla AWS offerings were somewhat limited when it came to monitoring models trained using RecordIO files (i.e. images), which hindered my goal of building a real-world MLOps pipeline.

To address these concerns and better align with my objective of creating a realistic MLOps pipeline, I have decided to switch to a text-based classification problem for my next project. Specifically, I will be working on sentiment analysis. This new project will involve a more real-world example of an ML pipeline, leveraging text data for sentiment classification. I will provide the GitHub link to the new project once it is ready for public access.

Despite the deprecation of this repository, I would like to highlight the valuable content it contains. The **exploration folder** includes IPython notebooks that delve into the data modeling aspect of the ML workflow, showcasing the transformation of the [Caltech-UCSD Birds-200-2011](https://www.kaggle.com/datasets/veeralakrishna/200-bird-species-with-11788-images) dataset into RecordIO files with train/validation/test split. These notebooks also cover the training and deployment of the object detection model, serving as a foundation for future image-based ML projects.

In addition, the **operations folder** provides a fully functional pipeline using the BIRDS-200 dataset, demonstrating the operational aspects of the ML workflow. The pipeline is designed as follows:

![BIRDS-200-pipeline](https://github.com/SanferD/birds-200-sagemaker-devops/assets/9338001/254c2916-8d08-4a1d-9182-144836943bce)

1. **Data Preprocessing:** The pipeline starts by downloading the dataset and performing necessary preparations. It creates a split into training, validation, and test sets. The training and validation data are saved as RecordIO files, while the test images are stored as JPEGs along with their corresponding labels in JSON format.

2. **Model Training:** The pipeline proceeds to train a VGG16 model using the prepared training and validation data. Note, the focus of this project is the pipeline, not the model.

3. **Model Creation:** Once the training is complete, the pipeline creates the model using the artifacts generated during the training step.

4. **Model Transformation:** The pipeline then moves on to applying batch transform using the trained model for inference. It fetches the raw test images in JPEG format, performs the transformation, and stores the model predictions in an S3 bucket.

5. **Evaluation:** Next, the pipeline evaluates the performance of the model. It compares the predictions generated in the previous step with the actual labels. The evaluation process includes computing the mean Average Precision (mAP) score, which provides a measure of the model's accuracy. The code implementation for this step is designed to be intuitive and easy to understand.

6. **Quality Check:** Following evaluation, the pipeline performs a quality check. It confirms whether the computed mAP score meets a specified threshold of 0.0. Note the focus of this project is the pipeline, not the model.

7. **Model Registration:** Finally, the pipeline registers the model in the SageMaker model registry, where it awaits approval. This step ensures proper management and versioning of the trained model.

Throughout the pipeline, each step carries out specific tasks autonomously to streamline the entire process of creating and evaluating the bird species identification model using AWS SageMaker.

I appreciate your understanding regarding the deprecation of this repository and hope that the upcoming sentiment analysis project will offer a more comprehensive example of an MLOps pipeline. Stay tuned for the GitHub link to the new project, and thank you for your interest in my work.

If you have any questions or concerns, please feel free to reach out to me.
