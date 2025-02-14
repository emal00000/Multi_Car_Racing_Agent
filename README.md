# Multi Car Racing
In this project, we explore the development and evaluation of the behaviors in multi-agent car racing. Our approach focuses on creating an abstraction of a complex multi-agent environment to be able to run it efficiently on mid-range PCs. Our abstraction focuses on reducing the state representation by focusing on a subset of the track that is represented as a sequence of points. Through this abstraction, we developed an agent capable of efficiently navigating the track, making significant progress toward the finish line.

## Environment Setup
1. Follow the installation instructions provided in the "multi_car_racing" repository linked below:
  - [Multi Car Racing](https://github.com/Sedwall/multi_car_racing)
2. Update line 17 in `requirements.txt` to point to the `multi_car_racing` directory cloned in the previous step.
3. Run the installation with the following command:
  ```bash
  pip install -r requirements.txt
  ```
###Resolving Box2D Compatibility Issue:
When attempting to run the model, you may encounter the following error: ```TypeError: in method 'b2RevoluteJoint___SetMotorSpeed', argument 2 of type 'float32'```
  To fix this, apply the following changes:
  1. In `box2d/car_dynamics.py`, line 145, replace:
     `w.joint.motorSpeed = dir*min(50.0*val, 3.0)`
     with
     `w.joint.motorSpeed = float(dir*min(50.0*val, 3.0))`
  3. In `box2d/car_dynamics.py`, line 213-215: replase:
        `w.ApplyForceToCenter( (
                p_force*side[0] + f_force*forw[0],
                p_force*side[1] + f_force*forw[1]), True )`
        with
        `w.ApplyForceToCenter( (
                float(p_force*side[0] + f_force*forw[0]),
                float(p_force*side[1] + f_force*forw[1]), True ))`
