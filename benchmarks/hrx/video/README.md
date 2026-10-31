# Video and image test set

This set checks visible content through a multimodal projector. The two
two-second MP4 clips are 256 × 256 pixels at four frames per second with no
audio. One shows a red circle followed by a blue circle; the other moves a
green square from left to right. The questions in [test_set.json](test_set.json)
ask for the first color and direction of movement. The dog photo is a still
image control with a yes/no question. Each question requests a single letter
so the worker can score it without a model judge.

The clips were drawn as RGB frames and encoded as H.264 with ffmpeg 7.0.2.
The dog photo was generated with the built-in imagegen tool, then resized to
512 pixels wide as JPEG. Its generation prompt requested one clearly
recognizable brown dog facing the camera on grass in daylight, with no other
animals, people, text, or watermark. All three fixtures are checked in so
benchmark results do not depend on an external media host.
