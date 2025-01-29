FROM busybox

RUN mkdir -p /tmp/knowledgeBase/

# OLS and AI models
ADD currentOlsSolution.pkl /tmp/knowledgeBase/
ADD currentAiSolution.h5 /tmp/knowledgeBase/

# License & readme
ADD README.md /tmp/knowledgeBase/
