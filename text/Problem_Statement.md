# Problem Statement

We want to build a software system that extracts statistical patterns from historic flight data and uses them to predict arrival delays of arbitrary future flights. As we build the that software system, we'll take particular note of the additional concepts that need to be understood by the involved software engineers. To me, a data engineer is a software engineer with a particular focus on data handling. She understands *big data* and how data it is used in the context of predictive functions.

*Note:* This subject has been discussed in various books and online courses, such as this *[coursera course on data engineering][CourseraGCP]* or Valiappa Lakshmannan's highly recommendable book *[Data Engineering on the Google Cloud Platform][Lakh01]*. But to my best knowledge it hasn't yet been studied from a rather academic software engineering point of view. We therefore believe that this tutorial add a valuable contribution to the discussions about data engineering, it's practitioners and their desired skills.

**A software Engineering Project:** To make a clear distinction, we start the project not as a machine learning but rather as a software engineering project. We'll collect requiremenents from the various stakeholders, devise an architecture based on those requirements. We consider quality metrics, deployment scenarios and maintenance aspects. 

**The traveller's problem:** 

**The airport planner's problem:**



**A first architecture sketch:** Imagine a web application that displays predicted arrival delays for every incoming flight of the next two weeks or so. This application may very well be just a new screen within the context of a larger work planning tool. Knowing that the predictions themselves come with particular problem characteristics, we decide to encapsulate the prediction in a RESTful service. Now, our application should reach out to that  service with a RESTful HTTP GET request like: *For this list of flights, give me your arrival delay predictions*. 

**Understanding the problem terminology:** Predicting a value $\mathbf{y}$, given some other values $\mathbf{x}$ may be understood as calling a special function $\mathbf{y} = f(\mathbf{x})$. That function is not known, but can be approximated by what we now know as *machine learning* (ML). In ML, that function is called the *model* or sometimes also the *hypothesis*. Without going into too much detail here, we note that there are various categories of models out there and they all come with parameters that need to be *trained*. Training is the process of feeding statistical data with known outcomes (*labels* $\mathbf{y}$) into the model and adapt its parameters again and again until the function values match the known outcome. If you haven't come across these concepts yet, I recommend Michael Nielson's popular introductory web book [Neural Networks and Deep Learning][Nielson]. It focusses on artificial neural networks, a particularly popular category of predictice models, but the concepts of *learning* or *training* apply to other models just the same way. 




[CourseraGCP]: https://www.coursera.org/specializations/gcp-data-machine-learning 

[Lakh01]: https://smile.amazon.de/gp/product/B0787L7RK3

[Nielson]: http://neuralnetworksanddeeplearning.com/