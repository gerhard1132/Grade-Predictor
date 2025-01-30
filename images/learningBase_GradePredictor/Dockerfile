FROM busybox

RUN mkdir -p /tmp/learningBase/
RUN mkdir -p /tmp/learningBase/train
RUN mkdir -p /tmp/learningBase/validation

# data
ADD train_data.csv /tmp/learningBase/train/
ADD test_data.csv /tmp/learningBase/validation/

# License & Readme
ADD Readme.md /tmp/learningBase/