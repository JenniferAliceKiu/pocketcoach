FROM metabase/metabase:latest

# Bind to the $PORT injected by Cloud Run
ENV MB_JETTY_HOST=0.0.0.0
# You can set a default, but $PORT will override at runtime
ENV MB_JETTY_PORT=${PORT:-3000}

# Use shell entrypoint so Docker expands ${PORT}
ENTRYPOINT ["sh","-c","java -DMB_JETTY_HOST=$MB_JETTY_HOST -DMB_JETTY_PORT=$MB_JETTY_PORT -jar /app/metabase.jar"]
