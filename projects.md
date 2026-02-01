---
layout: page
title: Projects
permalink: /projects/
---

Things I've built or am building.

<div class="projects-grid">
{% for project in site.projects %}
<div class="project-card">
  <h3><a href="{{ project.repo }}" target="_blank">{{ project.title }}</a></h3>
  <p>{{ project.description }}</p>
  <div class="project-tags">
    {% for tag in project.tags %}
    <span class="project-tag">{{ tag }}</span>
    {% endfor %}
  </div>
</div>
{% endfor %}
</div>
