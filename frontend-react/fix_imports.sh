#!/bin/bash

# Fix type imports in all files
sed -i "s/import { ReactNode }/import type { ReactNode }/" src/components/GlassCard.tsx
sed -i "s/import { ToastMessage }/import type { ToastMessage }/" src/components/Common/Toast.tsx
sed -i "s/import { CollectFormData, DataSource }/import type { CollectFormData, DataSource }/" src/components/DataCollection/CollectForm.tsx
sed -i "s/import { DataSource }/import type { DataSource }/" src/components/DataCollection/SourceSelector.tsx
sed -i "s/import { GraphData, GraphNode, GraphEdge }/import type { GraphData, GraphNode, GraphEdge }/" src/components/Graph/GraphVisualization.tsx
sed -i "s/import { QueryResponse }/import type { QueryResponse }/" src/components/Query/ResponseDisplay.tsx
sed -i "s/import { Session }/import type { Session }/" src/components/Sessions/SessionManager.tsx
sed -i "s/import { UploadedFile }/import type { UploadedFile }/" src/components/Upload/FileUpload.tsx
sed -i "s/import { VectorSearchResult }/import type { VectorSearchResult }/" src/components/Vector/VectorSearch.tsx
sed -i "s/import { Session }/import type { Session }/" src/pages/Sessions.tsx
sed -i "s/import { UploadedFile }/import type { UploadedFile }/" src/pages/Upload.tsx
sed -i "s/import axios, { AxiosInstance }/import axios, type { AxiosInstance }/" src/services/api.ts

# Remove UploadedFile from api.ts types import since it's not used there
sed -i "s/, *UploadedFile//" src/services/api.ts

# Remove unused useEffect import
sed -i "s/import { motion, useState, useEffect }/import { motion, useState }/" src/pages/Sessions.tsx

echo "Fixed all type imports"
