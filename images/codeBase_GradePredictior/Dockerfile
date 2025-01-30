FROM busybox

RUN mkdir -p /tmp/codeBase/
RUN mkdir -p /tmp/codeBase/ai_solution
RUN mkdir -p /tmp/codeBase/ols_solution

ADD ai_solution/ /tmp/codeBase/ai_solution
ADD ols_solution/ /tmp/codeBase/ols_solution

# License & readme
ADD README.md /tmp/codeBase/
