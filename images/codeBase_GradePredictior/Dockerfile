FROM python:3.12

ADD ai_solution/ /codeBase/ai_solution
ADD ols_solution/ /codeBase/ols_solution

# License & readme
ADD README.md /codeBase/

RUN pip3 install --no-cache-dir keras pandas statsmodels tensorflow
