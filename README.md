# MedicareLLMTraining

Training small LLM for a Voice Agent, what handles all future interactions with the Medicare Software in our future Mobile App.

The first Model will be trained to hadle 5 criterias:

- Select
- Insert
- Update
- Casual Talk Q&A
- Date Events

The sub models are trained on the special criteria of the first model one.

- To train the Models we use:

  [x] Tensorflow

  [x] Numpy

<p>
Within the file "train_", we define the training data — specifying which data the model should learn from and what classifications it should receive. In this case, the goal is to classify between: update, select, insert, cual talk, and appointment events. The script generates the corresponding files from this data, which we can then fine-tune to achieve up to 98.7% accuracy in providing the correct response.
</p>
<p>
In the image, you can see that the model achieves a hit rate of 98.42% on the first run. Without fine-tuning, it's not possible to get much higher.
</p>
<img src="images/train.png" alt="Train" width="600"/>
<p>
In the next step, we let the model run a test using the "test_" file. This shows us how well the model performs the tasks we expect of it. At the same time, for decisions where the model assigns itself a confidence level of less than 90%, it should display the corresponding sentences that were tested — so that we can include them in the fine-tuning process.
</p>
<p>
In the image, you can see that the model recognized "Einzelbetreuung" (one-on-one care) as "fixation wheelchair"
</p>
<img src="images/test.png" alt="Test" width="600"/>
<p>
During fine-tuning, the LLM is repeatedly trained with data until both the input data and the results reach 98% accuracy, allowing us to proceed to the audio testing phase.
</p>
<p>
In the following image, you can see an output from the fine-tuning test, where various natural-language sentences are passed to the LLM to evaluate its responses.
</p>
<img src="images/fine.png" alt="Fine" width="600"/>
<p>
Next, we test the real-world scenario using audio-to-text, to see whether the LLM provides appropriate responses.
</p>
<p>
It became evident that the speech-to-text implementation plays a crucial role in achieving good results from the LLM.
</p>
<p>
An example of this can be seen in the following image.
</p>
<img src="images/voice.png" alt="Fine" width="600"/>
<p>
Finally, we also evaluate the natural language processing to ensure that the extraction of the required data functions correctly.
</p>
<p>
An example of this can be seen in the following image.
</p>
<img src="images/nlp.png" alt="Fine" width="600"/>

<p>
The advantage of splitting into smaller models is the simple implementation on devices like smartphones, which do not have sufficient computing power for large models.
</p>
<p>
To fully guarantee data privacy at the same time, we use this approach of custom-built LLMs. This way, we achieve cost savings by avoiding third-party services, and data privacy is ensured since all processing happens entirely locally on the device.
</p>
