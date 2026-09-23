# --- build stage: produce a wheel -------------------------------------------
FROM python:3.12-slim AS build

WORKDIR /src
COPY pyproject.toml README.md LICENSE ./
COPY affordability ./affordability
RUN python -m pip install --upgrade pip build \
    && python -m build --wheel --outdir /dist

# --- runtime stage: non-root, wheel only ------------------------------------
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN groupadd --system scorer && useradd --system --gid scorer --create-home scorer

COPY --from=build /dist/*.whl /tmp/
RUN python -m pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Statements and payslips are mounted at /data; nothing is written back.
WORKDIR /data
USER scorer

ENTRYPOINT ["affordability"]
CMD ["--help"]
