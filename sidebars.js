/** @type {import('@docusaurus/plugin-content-docs').SidebarsConfig} */
const sidebars = {
  // tutorialSidebar 이름을 원하는 이름(예: wikiSidebar)으로 바꿀 수 있으나,
  // docusaurus.config.js에서도 동일하게 맞춰주어야 합니다. 여기서는 기본값을 사용합니다.
  tutorialSidebar: [
    // 1. 최상위 단일 문서 (Introduction)
    {
      type: 'doc',
      id: '00-introduction', // `docs/00-introduction.md` 파일을 가리킴
      label: '🚀 환영합니다',
    },

    // 2. 학습 및 스터디 카테고리
    {
      type: 'category',
      label: '📚 학습 및 스터디',
      link: {
        type: 'generated-index',
        description: '팀의 학습 활동과 커리큘럼을 정리합니다.',
      },
      items: [
        '01-learning/ml-study', // `docs/01-learning/ml-study.md`
        '01-learning/dl-study', // `docs/01-learning/dl-study.md`
      ],
    },

    // 3. 프로젝트 및 경진대회 카테고리
    {
      type: 'category',
      label: '🏆 프로젝트 및 경진대회',
      link: {
        type: 'generated-index',
        description: '실전 프로젝트와 AI 경진대회 참가 기록을 관리합니다.',
      },
      items: [
        '02-competitions/2025-power-usage-prediction', // `docs/02-competitions/2025-power-usage-prediction.md`
      ],
    },

    // 4. 핵심 연구 및 기술 카테고리
    {
      type: 'category',
      label: '🔬 핵심 연구 및 기술',
      link: {
        type: 'generated-index',
      },
      items: [
        '03-core-tech/quant-alpha',
        '03-core-tech/ai-model-architecture',
      ],
    },

    // 5. 팀 문화 및 운영 카테고리
    {
      type: 'category',
      label: '🤝 팀 문화 및 운영',
      link: {
        type: 'generated-index',
      },
      items: [
        '04-team/contribution-guide',
      ],
    },
  ],
};

module.exports = sidebars;