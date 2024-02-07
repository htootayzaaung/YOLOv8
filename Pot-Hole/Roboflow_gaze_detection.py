from roboflow import Roboflow

rf = Roboflow(api_key="Y1uEhWI1EFvz6IKyqn38")
project = rf.workspace().project("pothole-detection-yolov8-jcnms")
model = project.version("1").model

job_id, signed_url, expire_time = model.predict_video(
    "Videos/demo.mp4",
    fps=5,
    prediction_type="batch-video",
)

results = model.poll_until_video_results(job_id)

print(results)