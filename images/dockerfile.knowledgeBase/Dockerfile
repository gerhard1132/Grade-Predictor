# Nutze BusyBox als minimales Image
FROM busybox

# Erstelle den Ordner für das Modell
RUN mkdir -p /tmp/knowledgeBase/

# Füge das trainierte Modell hinzu
ADD currentOlsSolution.pkl /tmp/knowledgeBase/
# Falls du TensorFlow nutzt, ersetze `.pkl` mit `.keras`
# ADD currentSolutions.keras /tmp/knowledgeBase/

# Füge die Readme-Datei hinzu
ADD README.md /tmp/knowledgeBase/

# Standard-Kommando beim Start des Containers
CMD ["sh"]

