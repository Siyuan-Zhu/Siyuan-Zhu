FROM python:3.11-slim
WORKDIR /app
COPY requirements-ci.lock /app/
RUN pip install --no-cache-dir -c requirements-ci.lock httpx
COPY scripts/gcp_zeitgeist_bridge.py scripts/check_editorial_deadline.py /app/scripts/
COPY src/zeitgeist/edition_approval.py /app/src/zeitgeist/
ENTRYPOINT ["python", "scripts/gcp_zeitgeist_bridge.py"]
