# Healthcare & Life Sciences Demo

**AI-DLC Stage: Ideation**

Use case: Real World Evidence (RWE) - OMOP standardization, cohorting, comparative effectiveness

## Demo Flow

1. User describes RWE use case to ideation agent
2. Agent discovers UC tables (OMOP-style patient/condition/drug tables)
3. Agent generates spec for cohort building and analysis pipeline
4. Hand off to execution plane

## To Run Demo

```bash
# Start ideation conversation
lakefoundry ideate --pod-id hls-demo
```

Part of the Lakefoundry AI-DLC demo suite.
