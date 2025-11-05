{{/*
Expand the name of the chart.
*/}}
{{- define "researcherai.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "researcherai.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "researcherai.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "researcherai.labels" -}}
helm.sh/chart: {{ include "researcherai.chart" . }}
{{ include "researcherai.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "researcherai.selectorLabels" -}}
app.kubernetes.io/name: {{ include "researcherai.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "researcherai.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "researcherai.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Neo4j connection string
*/}}
{{- define "researcherai.neo4jUri" -}}
{{- if .Values.neo4j.enabled }}
bolt://{{ .Release.Name }}-neo4j:7687
{{- else }}
{{- .Values.neo4j.externalUri }}
{{- end }}
{{- end }}

{{/*
Qdrant connection string
*/}}
{{- define "researcherai.qdrantHost" -}}
{{- if .Values.qdrant.enabled }}
{{ .Release.Name }}-qdrant
{{- else }}
{{- .Values.qdrant.externalHost }}
{{- end }}
{{- end }}

{{/*
Kafka bootstrap servers
*/}}
{{- define "researcherai.kafkaBootstrap" -}}
{{- if .Values.kafka.enabled }}
{{ .Values.kafka.cluster.name }}-kafka-bootstrap:9092
{{- else }}
{{- .Values.kafka.externalBootstrap }}
{{- end }}
{{- end }}
