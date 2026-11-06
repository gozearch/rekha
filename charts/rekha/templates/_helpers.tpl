{{/*
Expand the name of the chart.
*/}}
{{- define "rekha.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
*/}}
{{- define "rekha.fullname" -}}
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
{{- define "rekha.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "rekha.labels" -}}
helm.sh/chart: {{ include "rekha.chart" . }}
{{ include "rekha.selectorLabels" . }}
{{- if .Values.image.tag }}
app.kubernetes.io/version: {{ .Values.image.tag | quote }}
{{- else }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "rekha.selectorLabels" -}}
app.kubernetes.io/name: {{ include "rekha.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Headless service name
*/}}
{{- define "rekha.headlessServiceName" -}}
{{- printf "%s-headless" (include "rekha.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Name of the Secret containing auth keys
*/}}
{{- define "rekha.secretName" -}}
{{- if .Values.auth.existingSecret }}
{{- .Values.auth.existingSecret }}
{{- else }}
{{- printf "%s-auth" (include "rekha.fullname" .) | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}

{{/*
Name of the ServiceAccount
*/}}
{{- define "rekha.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "rekha.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Compute comma-separated list of initial peers for cluster bootstrap.
Format: 1@pod-0.headless:port,2@pod-1.headless:port,...
*/}}
{{- define "rekha.initialPeers" -}}
{{- $fullname := include "rekha.fullname" . -}}
{{- $headless := include "rekha.headlessServiceName" . -}}
{{- $ns := .Release.Namespace -}}
{{- $port := .Values.ports.internal -}}
{{- $count := int .Values.cluster.initialClusterSize -}}
{{- $peers := list -}}
{{- range $i := until $count -}}
  {{- $id := add1 $i -}}
  {{- $peer := printf "%d@%s-%d.%s.%s.svc.cluster.local:%d" $id $fullname $i $headless $ns (int $port) -}}
  {{- $peers = append $peers $peer -}}
{{- end -}}
{{- join "," $peers -}}
{{- end }}
