# Nutze BusyBox als minimales Image
FROM busybox

# Erstelle den Ordner für die Aktivierungsdaten
RUN mkdir -p /tmp/activationBase/

# Füge die Aktivierungsdaten hinzu
ADD activationBase/activation_data.csv /tmp/activationBase/

# Füge die Readme-Datei hinzu
ADD activationBase/README.md /tmp/activationBase/

# Standard-Kommando beim Start des Containers
CMD ["sh"]


