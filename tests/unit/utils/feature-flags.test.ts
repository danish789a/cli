import { describe, expect, test } from 'vitest'

import { isFeatureFlagEnabled } from '../../../src/utils/feature-flags.js'

describe('isFeatureFlagEnabled', () => {
  test('should return true if feature flag is not present', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: true,
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('netlify_feature', siteInfo)

    expect(result).toBe(true)
  })

  test('should return true if feature flag is true', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: true,
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('cool_new_feature', siteInfo)

    expect(result).toBe(true)
  })

  test('should return true if feature flag is a string', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: 'my string',
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('cool_new_feature', siteInfo)

    expect(result).toBe(true)
  })

  test('should return true if feature flag is a number', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: 42,
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('cool_new_feature', siteInfo)

    expect(result).toBe(true)
  })

  test('should return true if feature flag is an object', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: { key: 'value' },
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('cool_new_feature', siteInfo)

    expect(result).toBe(true)
  })

  test('should return false if feature flag is false', async () => {
    const siteInfo = {
      feature_flags: {
        cool_new_feature: true,
        amazing_feature: false,
      },
    }

    // @ts-expect-error TS(2345) FIXME: Argument of type '{ feature_flags: { cool_new_feat... Remove this comment to see the full error message
    const result = isFeatureFlagEnabled('amazing_feature', siteInfo)

    expect(result).toBe(false)
  })
})
